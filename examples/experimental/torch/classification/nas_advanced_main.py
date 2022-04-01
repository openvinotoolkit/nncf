"""
 Copyright (c) 2022 Intel Corporation
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
import warnings
from pathlib import Path
from shutil import copyfile

import torch
import torch.nn as nn

from examples.torch.classification.main import create_data_loaders
from examples.torch.classification.main import create_datasets
from examples.torch.classification.main import get_argument_parser
from examples.torch.classification.main import inception_criterion_fn
from examples.torch.classification.main import train_epoch
from examples.torch.classification.main import validate
from examples.torch.common.argparser import parse_args
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import set_seed
from examples.torch.common.execution import start_worker
from examples.torch.common.model_loader import load_model
from examples.torch.common.sample_config import SampleConfig
from examples.torch.common.sample_config import create_sample_config
from examples.torch.common.utils import SafeMLFLow
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import configure_paths
from examples.torch.common.utils import create_code_snapshot
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.common.utils import print_args
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS.search import SearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.training import EpochBasedTrainingAlgorithm
from nncf.torch.initialization import default_criterion_fn
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.utils import is_main_process


def get_nas_argument_parser():
    parser = get_argument_parser()
    parser.add_argument('--train-steps', default=None, type=int,
                        help='Enables running training for the given number of steps')
    return parser


def get_optimizer(model, opt_config):
    def get_parameters(model, keys=None, mode='include'):
        if keys is None:
            for name, param in model.named_parameters():
                if param.requires_grad: yield param
        elif mode == 'include':
            for name, param in model.named_parameters():
                flag = False
                for key in keys:
                    if key in name:
                        flag = True
                        break
                if flag and param.requires_grad: yield param
        elif mode == 'exclude':
            for name, param in model.named_parameters():
                flag = True
                for key in keys:
                    if key in name:
                        flag = False
                        break
                if flag and param.requires_grad: yield param
        else:
            raise ValueError('do not support: %s' % mode)

    def build_optimizer(net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys):
        if no_decay_keys:
            assert isinstance(net_params, list) and len(net_params) == 2
            net_params = [
                {'params': net_params[0], 'weight_decay': weight_decay},
                {'params': net_params[1], 'weight_decay': 0},
            ]
        else:
            net_params = [{'params': net_params, 'weight_decay': weight_decay}]

        if opt_type == 'sgd':
            opt_param = {} if opt_param is None else opt_param
            momentum, nesterov = opt_param.get('momentum', 0.9), opt_param.get('nesterov', True)
            optimizer = torch.optim.SGD(net_params, init_lr, momentum=momentum, nesterov=nesterov)
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(net_params, init_lr)
        else:
            raise NotImplementedError
        return optimizer

    no_decay_keys = opt_config.no_decay_keys

    if no_decay_keys:
        keys = no_decay_keys.split('#')
        net_params = [
            get_parameters(model, keys, mode='exclude'),
            get_parameters(model, keys, mode='include'),
        ]
    else:
        # noinspection PyBroadException
        try:
            net_params = model.weight_parameters()
        except Exception:
            net_params = []
            for param in model.parameters():
                if param.requires_grad:
                    net_params.append(param)

    opt_type = opt_config.type
    opt_param = None
    init_lr = opt_config.base_lr
    weight_decay = opt_config.weight_decay

    optimizer = build_optimizer(net_params, opt_type, opt_param, init_lr, weight_decay, no_decay_keys)

    return optimizer


def label_smooth(target, num_classes: int, label_smoothing=0.1):
    batch_size = target.size(0)
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros((batch_size, num_classes), device=target.device)
    soft_target.scatter_(1, target, 1)
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / num_classes
    return soft_target


def cross_entropy_loss_with_soft_target(pred, soft_target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(- soft_target * logsoftmax(pred), 1))


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    soft_target = label_smooth(target, pred.size(1), label_smoothing)
    return cross_entropy_loss_with_soft_target(pred, soft_target)


def main(argv):
    parser = get_nas_argument_parser()
    args = parse_args(parser, argv)
    config = create_sample_config(args, parser)

    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config)
    copyfile(args.config, osp.join(config.log_dir, 'config.json'))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    config.execution_mode = get_execution_mode(config)

    start_worker(main_worker, config)


# pylint:disable=too-many-branches,too-many-statements
def main_worker(current_gpu, config: SampleConfig):
    configure_device(current_gpu, config)
    config.mlflow = SafeMLFLow(config)
    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    set_seed(config)

    opt_config = config.get('optimizer', {})

    # define loss function (criterion)
    if opt_config.label_smoothing:
        criterion = lambda pred, target: \
            cross_entropy_with_label_smoothing(pred, target, opt_config.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    model_name = config['model']
    train_criterion_fn = inception_criterion_fn if 'inception' in model_name else default_criterion_fn

    nncf_config = config.nncf_config
    pretrained = is_pretrained_model_requested(config)

    # Data loading code
    train_dataset, val_dataset = create_datasets(config)
    train_loader, _, val_loader, _ = create_data_loaders(config, train_dataset, val_dataset)

    bn_adapt_args = BNAdaptationInitArgs(data_loader=wrap_dataloader_for_init(train_loader), device=config.device)
    nncf_config.register_extra_structs([bn_adapt_args])
    # create model
    model = load_model(model_name,
                       pretrained=pretrained,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'),
                       weights_path=config.get('weights'))

    model.to(config.device)

    validate(val_loader, model, criterion, config)

    optimizer = get_optimizer(model, opt_config)

    def train_epoch_fn(loader, model_, compression_ctrl, epoch, optimizer_):
        train_epoch(loader, model_, criterion, train_criterion_fn, optimizer_, compression_ctrl, epoch, config,
                    train_iters=config.train_steps, log_training_info=True)

    def validate_model_fn(model_, loader):
        top1, top5, loss = validate(loader, model_, criterion, config, log_validation_info=False)
        return top1, top5, loss

    nncf_network = create_nncf_network(model, nncf_config)

    resuming_checkpoint_path = config.resuming_checkpoint_path
    if resuming_checkpoint_path is None:
        training_algorithm = EpochBasedTrainingAlgorithm.from_config(nncf_network, nncf_config)
    else:
        training_algorithm = EpochBasedTrainingAlgorithm.from_checkpoint(nncf_network, bn_adapt_args,
                                                                         resuming_checkpoint_path)

    if 'train' in config.mode:
        # Validate supernetwork
        top1, _, _ = validate(val_loader, model, criterion, config)

        nncf_network, elasticity_ctrl = training_algorithm.run(train_epoch_fn, train_loader,
                                                               validate_model_fn, val_loader, optimizer,
                                                               config.checkpoint_save_dir, config.tb,
                                                               config.train_steps)

        search_algo = SearchAlgorithm(model, elasticity_ctrl, nncf_config)

        elasticity_ctrl, best_config, performance_metrics = search_algo.run(validate_model_fn, val_loader, config.checkpoint_save_dir, tensorboard_writer=config.tb)

        print(best_config)
        print(performance_metrics)

    if 'test' in config.mode:
        validate(val_loader, model, criterion, config)


if __name__ == '__main__':
    main(sys.argv[1:])
