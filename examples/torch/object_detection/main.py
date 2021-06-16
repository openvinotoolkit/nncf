"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import os.path as osp
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import torch.utils.data as data

from examples.torch.common import restricted_pickle_module
from examples.torch.common.execution import set_seed
from examples.torch.common.model_loader import load_resuming_model_state_dict_and_checkpoint_from_path
from examples.torch.common.sample_config import create_sample_config, SampleConfig
from torch.optim.lr_scheduler import ReduceLROnPlateau

from examples.torch.common.argparser import get_common_argument_parser
from examples.torch.common.distributed import DistributedSampler
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution, start_worker
from nncf.api.compression import CompressionStage
from nncf.torch.initialization import register_default_init_args
from examples.torch.common.optimizer import get_parameter_groups, make_optimizer
from examples.torch.common.utils import get_name, make_additional_checkpoints, configure_paths, \
    create_code_snapshot, is_on_first_rank, configure_logging, print_args, is_pretrained_model_requested, \
    log_common_mlflow_params, SafeMLFLow, configure_device
from examples.torch.common.utils import write_metrics
from examples.torch.object_detection.dataset import detection_collate, get_testing_dataset, get_training_dataset
from examples.torch.object_detection.eval import test_net
from examples.torch.object_detection.layers.modules import MultiBoxLoss
from examples.torch.object_detection.model import build_ssd
from nncf.torch import create_compressed_model
from nncf.torch import load_state
from nncf.torch.dynamic_graph.graph_tracer import create_input_infos
from nncf.torch.utils import is_main_process


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_option(args, config, key, default=None):
    """Gets key option from args if it is provided, otherwise tries to get it from config"""
    if hasattr(args, key) and getattr(args, key) is not None:
        return getattr(args, key)
    return config.get(key, default)


def get_argument_parser():
    parser = get_common_argument_parser()

    parser.add_argument('--basenet', default='', help='pretrained base model, should be located in save_folder')
    parser.add_argument('--test-interval', default=5000, type=int, help='test interval')
    parser.add_argument("--dataset", help="Dataset to use.", choices=["voc", "coco"], default=None)
    parser.add_argument('--train_imgs', help='path to training images or VOC root directory')
    parser.add_argument('--train_anno', help='path to training annotations or VOC root directory')
    parser.add_argument('--test_imgs', help='path to testing images or VOC root directory')
    parser.add_argument('--test_anno', help='path to testing annotations or VOC root directory')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
    config = create_sample_config(args, parser)

    configure_paths(config)
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    config.execution_mode = get_execution_mode(config)

    if config.dataset_dir is not None:
        config.train_imgs = config.train_anno = config.test_imgs = config.test_anno = config.dataset_dir
    start_worker(main_worker, config)


# pylint:disable=too-many-branches
def main_worker(current_gpu, config):
    #################################
    # Setup experiment environment
    #################################
    configure_device(current_gpu, config)
    config.mlflow = SafeMLFLow(config)
    if is_on_first_rank(config):
        configure_logging(logger, config)
        print_args(config)

    set_seed(config)

    config.start_iter = 0
    nncf_config = config.nncf_config
    ##########################
    # Prepare metrics log file
    ##########################

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    ###########################
    # Criterion
    ###########################

    criterion = MultiBoxLoss(
        config,
        config['num_classes'],
        overlap_thresh=0.5,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=3,
        neg_overlap=0.5,
        encode_target=False,
        device=config.device
    )

    train_data_loader = test_data_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path

    ###########################
    # Prepare data
    ###########################

    pretrained = is_pretrained_model_requested(config)

    if config.to_onnx is not None:
        assert pretrained or (resuming_checkpoint_path is not None)
    else:
        test_data_loader, train_data_loader, init_data_loader = create_dataloaders(config)

        def criterion_fn(model_outputs, target, criterion):
            loss_l, loss_c = criterion(model_outputs, target)
            return loss_l + loss_c

        def autoq_test_fn(model, eval_loader):
            # RL is maximization, change the loss polarity
            return -1 * test_net(model, config.device, eval_loader, distributed=config.distributed,
                                 loss_inference=True, criterion=criterion)

        nncf_config = register_default_init_args(
            nncf_config, init_data_loader, criterion, criterion_fn,
            autoq_test_fn, test_data_loader, config.device)

    ##################
    # Prepare model
    ##################
    resuming_checkpoint_path = config.resuming_checkpoint_path

    resuming_model_sd = None
    if resuming_checkpoint_path is not None:
        resuming_model_sd, resuming_checkpoint = load_resuming_model_state_dict_and_checkpoint_from_path(
            resuming_checkpoint_path)

    compression_ctrl, net = create_model(config, resuming_model_sd)
    if config.distributed:
        config.batch_size //= config.ngpus_per_node
        config.workers //= config.ngpus_per_node
        compression_ctrl.distributed()

    ###########################
    # Optimizer
    ###########################

    params_to_optimize = get_parameter_groups(net, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

    #################################
    # Load additional checkpoint data
    #################################

    if resuming_checkpoint_path is not None and config.mode.lower() == 'train' and config.to_onnx is None:
        compression_ctrl.load_state(resuming_checkpoint)
        optimizer.load_state_dict(resuming_checkpoint.get('optimizer', optimizer.state_dict()))
        config.start_iter = resuming_checkpoint.get('iter', 0) + 1

    log_common_mlflow_params(config)

    if config.to_onnx:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    if is_main_process():
        statistics = compression_ctrl.statistics()
        logger.info(statistics.to_str())

    if config.mode.lower() == 'test':
        with torch.no_grad():
            net.eval()
            if config['ssd_params'].get('loss_inference', False):
                model_loss = test_net(net, config.device, test_data_loader, distributed=config.distributed,
                                      loss_inference=True, criterion=criterion)
                logger.info("Final model loss: {:.3f}".format(model_loss))
            else:
                mAp = test_net(net, config.device, test_data_loader, distributed=config.distributed)
                if config.metrics_dump is not None:
                    write_metrics(mAp, config.metrics_dump)
            return

    train(net, compression_ctrl, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler)


def create_dataloaders(config):
    logger.info('Loading Dataset...')
    train_dataset = get_training_dataset(config.dataset, config.train_anno, config.train_imgs, config)
    logger.info("Loaded {} training images".format(len(train_dataset)))
    if config.distributed:
        sampler_seed = 0 if config.seed is None else config.seed
        dist_sampler_shuffle = config.seed is None
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        num_replicas=config.ngpus_per_node,
                                                                        rank=config.rank,
                                                                        seed=sampler_seed,
                                                                        shuffle=dist_sampler_shuffle)
    else:
        train_sampler = None

    train_shuffle = train_sampler is None and config.seed is None

    def create_train_data_loader(batch_size):
        return data.DataLoader(
            train_dataset, batch_size,
            num_workers=config.workers,
            shuffle=train_shuffle,
            collate_fn=detection_collate,
            pin_memory=True,
            sampler=train_sampler
        )

    train_data_loader = create_train_data_loader(config.batch_size)
    if config.batch_size_init:
        init_data_loader = create_train_data_loader(config.batch_size_init)
    else:
        init_data_loader = deepcopy(train_data_loader)
    if config.distributed:
        init_data_loader.num_workers = 0  # PyTorch multiprocessing dataloader issue WA

    test_dataset = get_testing_dataset(config.dataset, config.test_anno, config.test_imgs, config)
    logger.info("Loaded {} testing images".format(len(test_dataset)))
    if config.distributed:
        test_sampler = DistributedSampler(test_dataset, config.rank, config.world_size)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_data_loader = data.DataLoader(
        test_dataset, config.batch_size,
        num_workers=config.workers,
        shuffle=False,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler
    )
    return test_data_loader, train_data_loader, init_data_loader


def create_model(config: SampleConfig, resuming_model_sd: dict = None):
    input_info_list = create_input_infos(config.nncf_config)
    image_size = input_info_list[0].shape[-1]
    ssd_net = build_ssd(config.model, config.ssd_params, image_size, config.num_classes, config)
    weights = config.get('weights')
    if weights:
        sd = torch.load(weights, map_location='cpu',
                        pickle_module=restricted_pickle_module)
        load_state(ssd_net, sd)

    ssd_net.to(config.device)

    compression_ctrl, compressed_model = create_compressed_model(ssd_net, config.nncf_config, resuming_model_sd)
    compressed_model, _ = prepare_model_for_execution(compressed_model, config)

    compressed_model.train()
    return compression_ctrl, compressed_model


def train_step(batch_iterator, compression_ctrl, config, criterion, net, train_data_loader):
    batch_loss_l = torch.tensor(0.).to(config.device)
    batch_loss_c = torch.tensor(0.).to(config.device)
    batch_loss = torch.tensor(0.).to(config.device)
    for _ in range(0, config.iter_size):
        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            logger.debug("StopIteration: can not load batch")
            batch_iterator = iter(train_data_loader)
            break

        images = images.to(config.device)
        targets = [anno.requires_grad_(False).to(config.device) for anno in targets]

        # forward
        out = net(images)
        # backprop
        loss_l, loss_c = criterion(out, targets)
        loss_comp = compression_ctrl.loss()
        loss = loss_l + loss_c + loss_comp
        batch_loss += loss
        loss.backward()
        batch_loss_l += loss_l
        batch_loss_c += loss_c
    return batch_iterator, batch_loss, batch_loss_c, batch_loss_l, loss_comp


# pylint: disable=too-many-statements
def train(net, compression_ctrl, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler):
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0

    epoch_size = len(train_data_loader)
    logger.info('Training {} on {} dataset...'.format(config.model, train_data_loader.dataset.name))
    batch_iterator = None

    t_start = time.time()

    best_mAp = 0
    best_compression_stage = CompressionStage.UNCOMPRESSED
    test_freq_in_epochs = max(config.test_interval // epoch_size, 1)

    for iteration in range(config.start_iter, config['max_iter']):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(train_data_loader)

        epoch = iteration // epoch_size

        compression_ctrl.scheduler.step()
        if iteration % epoch_size == 0:
            compression_ctrl.scheduler.epoch_step(epoch)
            if is_main_process():
                statistics = compression_ctrl.statistics()
                logger.info(statistics.to_str())

        if (iteration + 1) % epoch_size == 0:
            compression_stage = compression_ctrl.compression_stage()
            is_best = False
            if (epoch + 1) % test_freq_in_epochs == 0:
                with torch.no_grad():
                    net.eval()
                    mAP = test_net(net, config.device, test_data_loader, distributed=config.multiprocessing_distributed)
                    is_best_by_mAP = mAP > best_mAp and compression_stage == best_compression_stage
                    is_best = is_best_by_mAP or compression_stage > best_compression_stage
                    if is_best:
                        best_mAp = mAP
                    best_compression_stage = max(compression_stage, best_compression_stage)
                    if isinstance(lr_scheduler, ReduceLROnPlateau):
                        lr_scheduler.step(mAP)
                    net.train()

            if is_on_first_rank(config):
                logger.info('Saving state, iter: {}'.format(iteration))

                checkpoint_file_path = osp.join(config.checkpoint_save_dir, "{}_last.pth".format(get_name(config)))
                torch.save({
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iter': iteration,
                    'scheduler': compression_ctrl.scheduler.get_state(),
                    'compression_stage': compression_stage,
                }, str(checkpoint_file_path))
                make_additional_checkpoints(checkpoint_file_path,
                                            is_best=is_best,
                                            epoch=epoch + 1,
                                            config=config)

            # Learning rate scheduling should be applied after optimizer’s update
            if not isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(epoch)

        optimizer.zero_grad()
        batch_iterator, batch_loss, batch_loss_c, batch_loss_l, loss_comp = train_step(
            batch_iterator, compression_ctrl, config, criterion, net, train_data_loader
        )
        optimizer.step()

        batch_loss_l = batch_loss_l / config.iter_size
        batch_loss_c = batch_loss_c / config.iter_size
        model_loss = (batch_loss_l + batch_loss_c) / config.iter_size
        batch_loss = batch_loss / config.iter_size

        loc_loss += batch_loss_l.item()
        conf_loss += batch_loss_c.item()

        ###########################
        # Logging
        ###########################

        if is_on_first_rank(config):
            config.tb.add_scalar("train/loss_l", batch_loss_l.item(), iteration)
            config.tb.add_scalar("train/loss_c", batch_loss_c.item(), iteration)
            config.tb.add_scalar("train/loss", batch_loss.item(), iteration)

        if iteration % config.print_freq == 0:
            t_finish = time.time()
            t_elapsed = t_finish - t_start
            t_start = time.time()
            logger.info('{}: iter {} epoch {} || Loss: {:.4} || Time {:.4}s || lr: {} || CR loss: {}'.format(
                config.rank, iteration, epoch, model_loss.item(), t_elapsed, optimizer.param_groups[0]['lr'],
                loss_comp.item() if isinstance(loss_comp, torch.Tensor) else loss_comp
            ))

    if config.metrics_dump is not None:
        write_metrics(best_mAp, config.metrics_dump)


if __name__ == '__main__':
    main(sys.argv[1:])
