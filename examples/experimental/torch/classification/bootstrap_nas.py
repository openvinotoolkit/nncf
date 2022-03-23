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

from torch import nn

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
from examples.torch.common.optimizer import get_parameter_groups
from examples.torch.common.optimizer import make_optimizer
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
from nncf.experimental.torch.nas.bootstrapNAS import EpochBasedTrainingAlgorithm
from nncf.torch.initialization import default_criterion_fn
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.utils import is_main_process


def get_nas_argument_parser():
    parser = get_argument_parser()
    parser.add_argument('--train-steps', default=None, type=int,
                        help='Enables running training for the given number of steps')
    return parser


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

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

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

    # define optimizer
    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)

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
        training_algorithm.run(train_epoch_fn, train_loader, lr_scheduler,
                               validate_model_fn, val_loader, optimizer,
                               config.checkpoint_save_dir, config.tb)

    if 'test' in config.mode:
        validate(val_loader, model, criterion, config)


if __name__ == '__main__':
    main(sys.argv[1:])
