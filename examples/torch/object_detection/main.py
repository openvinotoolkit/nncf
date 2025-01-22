# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os.path as osp
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils import data

from examples.common.paths import configure_paths
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.common import restricted_pickle_module
from examples.torch.common.argparser import get_common_argument_parser
from examples.torch.common.argparser import parse_args
from examples.torch.common.distributed import DistributedSampler
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.execution import set_seed
from examples.torch.common.execution import start_worker
from examples.torch.common.export import export_model
from examples.torch.common.model_loader import COMPRESSION_STATE_ATTR
from examples.torch.common.model_loader import MODEL_STATE_ATTR
from examples.torch.common.model_loader import extract_model_and_compression_states
from examples.torch.common.model_loader import load_resuming_checkpoint
from examples.torch.common.optimizer import get_parameter_groups
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import create_code_snapshot
from examples.torch.common.utils import get_run_name
from examples.torch.common.utils import is_on_first_rank
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.common.utils import make_additional_checkpoints
from examples.torch.common.utils import print_args
from examples.torch.common.utils import write_metrics
from examples.torch.object_detection.dataset import detection_collate
from examples.torch.object_detection.dataset import get_testing_dataset
from examples.torch.object_detection.dataset import get_training_dataset
from examples.torch.object_detection.eval import test_net
from examples.torch.object_detection.layers.modules import MultiBoxLoss
from examples.torch.object_detection.model import build_ssd
from nncf.api.compression import CompressionStage
from nncf.common.accuracy_aware_training import create_accuracy_aware_training_loop
from nncf.config.structures import ModelEvaluationArgs
from nncf.config.utils import is_accuracy_aware_training
from nncf.torch import create_compressed_model
from nncf.torch import load_state
from nncf.torch.dynamic_graph.io_handling import FillerInputInfo
from nncf.torch.initialization import register_default_init_args
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

    parser.add_argument("--basenet", default="", help="pretrained base model, should be located in save_folder")
    parser.add_argument("--test-interval", default=1, type=int, help="test interval")
    parser.add_argument("--dataset", help="Dataset to use.", choices=["voc", "coco"], default=None)
    parser.add_argument("--train_imgs", help="path to training images or VOC root directory")
    parser.add_argument("--train_anno", help="path to training annotations or VOC root directory")
    parser.add_argument("--test_imgs", help="path to testing images or VOC root directory")
    parser.add_argument("--test_anno", help="path to testing annotations or VOC root directory")
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parse_args(parser, argv)
    config = create_sample_config(args, parser)

    configure_paths(config, get_run_name(config))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    config.execution_mode = get_execution_mode(config)

    if config.dataset_dir is not None:
        config.train_imgs = config.train_anno = config.test_imgs = config.test_anno = config.dataset_dir
    start_worker(main_worker, config)


def main_worker(current_gpu, config):
    #################################
    # Setup experiment environment
    #################################
    configure_device(current_gpu, config)
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
        config["num_classes"],
        overlap_thresh=0.5,
        prior_for_matching=True,
        bkg_label=0,
        neg_mining=True,
        neg_pos=3,
        neg_overlap=0.5,
        encode_target=False,
        device=config.device,
    )

    train_data_loader = test_data_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path

    ###########################
    # Prepare data
    ###########################

    pretrained = is_pretrained_model_requested(config)

    is_export_only = "export" in config.mode and ("train" not in config.mode and "test" not in config.mode)
    if is_export_only:
        assert pretrained or (resuming_checkpoint_path is not None)
    else:
        test_data_loader, train_data_loader, init_data_loader = create_dataloaders(config)

        def criterion_fn(model_outputs, target, criterion):
            loss_l, loss_c = criterion(model_outputs, target)
            return loss_l + loss_c

        def autoq_test_fn(model, eval_loader):
            # RL is maximization, change the loss polarity
            return -1 * test_net(
                model,
                config.device,
                eval_loader,
                distributed=config.distributed,
                loss_inference=True,
                criterion=criterion,
            )

        def model_eval_fn(model):
            model.eval()
            mAP = test_net(model, config.device, test_data_loader, distributed=config.distributed, criterion=criterion)
            return mAP

        nncf_config = register_default_init_args(
            nncf_config,
            init_data_loader,
            criterion=criterion,
            criterion_fn=criterion_fn,
            autoq_eval_fn=autoq_test_fn,
            val_loader=test_data_loader,
            model_eval_fn=model_eval_fn,
            device=config.device,
        )

    ##################
    # Prepare model
    ##################
    resuming_checkpoint_path = config.resuming_checkpoint_path

    resuming_checkpoint = None
    if resuming_checkpoint_path is not None:
        resuming_checkpoint = load_resuming_checkpoint(resuming_checkpoint_path)
    net = create_model(config)
    if "train" in config.mode and is_accuracy_aware_training(config):
        with torch.no_grad():
            uncompressed_model_accuracy = config.nncf_config.get_extra_struct(ModelEvaluationArgs).eval_fn(net)
    compression_ctrl, net = compress_model(net, config, resuming_checkpoint)
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

    if resuming_checkpoint_path is not None and "train" in config.mode:
        optimizer.load_state_dict(resuming_checkpoint.get("optimizer", optimizer.state_dict()))
        config.start_epoch = resuming_checkpoint.get("epoch", 0) + 1

    if is_export_only:
        export_model(compression_ctrl, config)
        return

    if is_main_process():
        statistics = compression_ctrl.statistics()
        logger.info(statistics.to_str())

    if "train" in config.mode and is_accuracy_aware_training(config):
        # validation function that returns the target metric value

        def validate_fn(model, epoch):
            model.eval()
            mAP = test_net(model, config.device, test_data_loader, distributed=config.distributed)
            model.train()
            return mAP

        # training function that trains the model for one epoch (full training dataset pass)
        # it is assumed that all the NNCF-related methods are properly called inside of
        # this function (like e.g. the step and epoch_step methods of the compression scheduler)
        def train_epoch_fn(compression_ctrl, model, epoch, optimizer, **kwargs):
            loc_loss = 0
            conf_loss = 0
            epoch_size = len(train_data_loader)
            train_epoch(
                compression_ctrl,
                model,
                config,
                train_data_loader,
                criterion,
                optimizer,
                epoch_size,
                epoch,
                loc_loss,
                conf_loss,
            )

        # function that initializes optimizers & lr schedulers to start training
        def configure_optimizers_fn():
            params_to_optimize = get_parameter_groups(net, config)
            optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)
            return optimizer, lr_scheduler

        acc_aware_training_loop = create_accuracy_aware_training_loop(
            nncf_config, compression_ctrl, uncompressed_model_accuracy
        )
        net = acc_aware_training_loop.run(
            net,
            train_epoch_fn=train_epoch_fn,
            validate_fn=validate_fn,
            configure_optimizers_fn=configure_optimizers_fn,
            tensorboard_writer=config.tb,
            log_dir=config.log_dir,
        )
        logger.info(f"Compressed model statistics:\n{acc_aware_training_loop.statistics.to_str()}")
    elif "train" in config.mode:
        train(net, compression_ctrl, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler)

    if "test" in config.mode:
        with torch.no_grad():
            val_net = net
            net.eval()
            if config["ssd_params"].get("loss_inference", False):
                model_loss = test_net(
                    val_net,
                    config.device,
                    test_data_loader,
                    distributed=config.distributed,
                    loss_inference=True,
                    criterion=criterion,
                )
                logger.info("Final model loss: {:.3f}".format(model_loss))
            else:
                mAp = test_net(val_net, config.device, test_data_loader, distributed=config.distributed)
                if config.metrics_dump is not None:
                    write_metrics(mAp, config.metrics_dump)

    if "export" in config.mode:
        export_model(compression_ctrl, config)


def create_dataloaders(config):
    logger.info("Loading Dataset...")
    train_dataset = get_training_dataset(config.dataset, config.train_anno, config.train_imgs, config)
    logger.info("Loaded {} training images".format(len(train_dataset)))
    if config.distributed:
        sampler_seed = 0 if config.seed is None else config.seed
        dist_sampler_shuffle = config.seed is None
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=config.ngpus_per_node,
            rank=config.rank,
            seed=sampler_seed,
            shuffle=dist_sampler_shuffle,
        )
    else:
        train_sampler = None

    train_shuffle = train_sampler is None and config.seed is None

    def create_train_data_loader(batch_size):
        return data.DataLoader(
            train_dataset,
            batch_size,
            num_workers=config.workers,
            shuffle=train_shuffle,
            collate_fn=detection_collate,
            pin_memory=True,
            sampler=train_sampler,
        )

    train_data_loader = create_train_data_loader(config.batch_size)
    if config.batch_size_init:
        init_data_loader = create_train_data_loader(config.batch_size_init)
    else:
        init_data_loader = deepcopy(train_data_loader)

    test_dataset = get_testing_dataset(config.dataset, config.test_anno, config.test_imgs, config)
    logger.info("Loaded {} testing images".format(len(test_dataset)))
    if config.distributed:
        test_sampler = DistributedSampler(test_dataset, config.rank, config.world_size)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    test_data_loader = data.DataLoader(
        test_dataset,
        config.batch_size,
        num_workers=config.workers,
        shuffle=False,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler,
    )
    return test_data_loader, train_data_loader, init_data_loader


def create_model(config: SampleConfig):
    input_info = FillerInputInfo.from_nncf_config(config.nncf_config)
    image_size = input_info.elements[0].shape[-1]
    ssd_net = build_ssd(config.model, config.ssd_params, image_size, config.num_classes, config)
    weights = config.get("weights")
    if weights:
        sd = torch.load(weights, map_location="cpu", pickle_module=restricted_pickle_module)
        sd = sd["state_dict"]
        load_state(ssd_net, sd)

    ssd_net.to(config.device)

    return ssd_net


def compress_model(model: torch.nn.Module, config: SampleConfig, resuming_checkpoint: dict = None):
    model_state_dict, compression_state = extract_model_and_compression_states(resuming_checkpoint)
    compression_ctrl, compressed_model = create_compressed_model(model, config.nncf_config, compression_state)
    if model_state_dict is not None:
        load_state(compressed_model, model_state_dict, is_resume=True)
    compressed_model, _ = prepare_model_for_execution(compressed_model, config)

    compressed_model.train()
    return compression_ctrl, compressed_model


def train_step(batch_iterator, compression_ctrl, config, criterion, net, train_data_loader):
    batch_loss_l = torch.tensor(0.0).to(config.device)
    batch_loss_c = torch.tensor(0.0).to(config.device)
    batch_loss = torch.tensor(0.0).to(config.device)
    # load train data
    try:
        images, targets = next(batch_iterator)
    except StopIteration:
        logger.debug("StopIteration: can not load batch")
        batch_iterator = iter(train_data_loader)

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


def train(net, compression_ctrl, train_data_loader, test_data_loader, criterion, optimizer, config, lr_scheduler):
    net.train()
    loc_loss = 0
    conf_loss = 0

    epoch_size = len(train_data_loader)
    logger.info("Training {} on {} dataset...".format(config.model, train_data_loader.dataset.name))

    best_mAp = 0
    best_compression_stage = CompressionStage.UNCOMPRESSED
    test_freq_in_epochs = config.test_interval
    if config.test_interval is None:
        test_freq_in_epochs = 1

    max_epochs = config["epochs"]

    for epoch in range(config.start_epoch, max_epochs):
        compression_ctrl.scheduler.epoch_step(epoch)

        train_epoch(
            compression_ctrl,
            net,
            config,
            train_data_loader,
            criterion,
            optimizer,
            epoch_size,
            epoch,
            loc_loss,
            conf_loss,
        )

        if is_main_process():
            logger.info(compression_ctrl.statistics().to_str())

        compression_stage = compression_ctrl.compression_stage()
        is_best = False
        if (epoch + 1) % test_freq_in_epochs == 0:
            with torch.no_grad():
                net.eval()
                mAP = test_net(net, config.device, test_data_loader, distributed=config.multiprocessing_distributed)
                config.tb.add_scalar("eval/mAp", mAP, epoch)
                is_best_by_mAP = mAP > best_mAp and compression_stage == best_compression_stage
                is_best = is_best_by_mAP or compression_stage > best_compression_stage
                if is_best:
                    best_mAp = mAP
                best_compression_stage = max(compression_stage, best_compression_stage)
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(mAP)
                net.train()

        if is_on_first_rank(config):
            logger.info("Saving state, epoch: {}".format(epoch))

            checkpoint_file_path = osp.join(config.checkpoint_save_dir, "{}_last.pth".format(get_run_name(config)))
            torch.save(
                {
                    MODEL_STATE_ATTR: net.state_dict(),
                    COMPRESSION_STATE_ATTR: compression_ctrl.get_compression_state(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                },
                str(checkpoint_file_path),
            )
            make_additional_checkpoints(checkpoint_file_path, is_best=is_best, epoch=epoch + 1, config=config)

        # Learning rate scheduling should be applied after optimizer’s update
        if not isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(epoch)

        compression_ctrl.scheduler.epoch_step(epoch)

    if config.metrics_dump is not None:
        write_metrics(best_mAp, config.metrics_dump)


def train_epoch(
    compression_ctrl, net, config, train_data_loader, criterion, optimizer, epoch_size, epoch, loc_loss, conf_loss
):
    batch_iterator = iter(train_data_loader)
    t_start = time.time()

    for iteration in range(0, epoch_size):
        compression_ctrl.scheduler.step()
        optimizer.zero_grad()
        batch_iterator, batch_loss, batch_loss_c, batch_loss_l, loss_comp = train_step(
            batch_iterator, compression_ctrl, config, criterion, net, train_data_loader
        )
        optimizer.step()

        model_loss = batch_loss_l + batch_loss_c

        loc_loss += batch_loss_l.item()
        conf_loss += batch_loss_c.item()

        if is_on_first_rank(config):
            config.tb.add_scalar("train/loss_l", batch_loss_l.item(), iteration + epoch_size * epoch)
            config.tb.add_scalar("train/loss_c", batch_loss_c.item(), iteration + epoch_size * epoch)
            config.tb.add_scalar("train/loss", batch_loss.item(), iteration + epoch_size * epoch)

        if iteration % config.print_freq == 0:
            t_finish = time.time()
            t_elapsed = t_finish - t_start
            t_start = time.time()
            logger.info(
                "{}: iter {} epoch {} || Loss: {:.4} || Time {:.4}s || lr: {} || CR loss: {}".format(
                    config.rank,
                    iteration,
                    epoch,
                    model_loss.item(),
                    t_elapsed,
                    optimizer.param_groups[0]["lr"],
                    loss_comp.item() if isinstance(loss_comp, torch.Tensor) else loss_comp,
                )
            )


if __name__ == "__main__":
    main(sys.argv[1:])
