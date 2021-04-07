"""
 Copyright (c) 2021 Intel Corporation
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
from pathlib import Path

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.models as models
import warnings
from shutil import copyfile

from examples.classification.main import create_data_loaders, validate, create_datasets, \
    inception_criterion_fn, train_epoch
from examples.common.argparser import get_common_argument_parser
from examples.common.example_logger import logger
from examples.common.execution import ExecutionMode, get_execution_mode, \
    prepare_model_for_execution, start_worker
from examples.common.model_loader import load_model
from examples.common.optimizer import get_parameter_groups, make_optimizer
from examples.common.sample_config import SampleConfig, create_sample_config
from examples.common.utils import configure_logging, configure_paths, create_code_snapshot, \
    print_args, is_staged_quantization, print_statistics, \
    is_pretrained_model_requested, log_common_mlflow_params, SafeMLFLow, configure_device
from examples.common.utils import write_metrics
from nncf import create_compressed_model
from nncf import run_accuracy_aware_compressed_training, PTAccuracyAwareTrainingRunner
from nncf.initialization import register_default_init_args
from nncf.initialization import register_training_loop_args
from nncf.initialization import default_criterion_fn
from nncf.utils import is_main_process
from examples.classification.common import set_seed, load_resuming_checkpoint

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

def get_argument_parser():
    parser = get_common_argument_parser()
    parser.add_argument(
        "--dataset",
        help="Dataset to use.",
        choices=["imagenet", "cifar100", "cifar10"],
        default=None
    )
    parser.add_argument('--test-every-n-epochs', default=1, type=int,
                        help='Enables running validation every given number of epochs')
    return parser


def main(argv):
    parser = get_argument_parser()
    args = parser.parse_args(args=argv)
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

    if config.metrics_dump is not None:
        write_metrics(0, config.metrics_dump)

    if not is_staged_quantization(config):
        start_worker(main_worker, config)
    else:
        from examples.classification.staged_quantization_worker import staged_quantization_main_worker
        start_worker(staged_quantization_main_worker, config)


# pylint:disable=too-many-branches
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

    train_loader = val_loader = None
    resuming_checkpoint_path = config.resuming_checkpoint_path
    nncf_config = config.nncf_config
    pretrained = is_pretrained_model_requested(config)

    if config.to_onnx is not None:
        assert pretrained or (resuming_checkpoint_path is not None)
    else:
        # Data loading code
        train_dataset, val_dataset = create_datasets(config)
        train_loader, _, val_loader, init_loader = create_data_loaders(config, train_dataset, val_dataset)

        def evaluation_fn(model, eval_loader):
            top1, _ = validate(eval_loader, model, criterion, config)
            return top1

        nncf_config = register_default_init_args(
            nncf_config, init_loader, criterion, train_criterion_fn,
            evaluation_fn, val_loader, config.device)

    # create model
    model = load_model(model_name,
                       pretrained=pretrained,
                       num_classes=config.get('num_classes', 1000),
                       model_params=config.get('model_params'),
                       weights_path=config.get('weights'))

    model.to(config.device)

    resuming_model_sd, _ = load_resuming_checkpoint(resuming_checkpoint_path)
    compression_ctrl, model = create_compressed_model(model, nncf_config,
                                                      resuming_state_dict=resuming_model_sd,
                                                      should_eval_original_model=True)

    if config.to_onnx:
        compression_ctrl.export_model(config.to_onnx)
        logger.info("Saved to {}".format(config.to_onnx))
        return

    model, _ = prepare_model_for_execution(model, config)
    if config.distributed:
        compression_ctrl.distributed()

    log_common_mlflow_params(config)

    if config.execution_mode != ExecutionMode.CPU_ONLY:
        cudnn.benchmark = True

    if is_main_process():
        print_statistics(compression_ctrl.statistics())

    if config.mode.lower() == 'test':
        validate(val_loader, model, criterion, config)

    if config.mode.lower() == 'train':

        # validation function that returns the target metric value
        # pylint: disable=E1123
        def validate_fn(model, epoch):
            top1, _ = validate(val_loader, model, criterion, config, epoch=epoch)
            return top1

        # training function that trains the model for one epoch (full training dataset pass)
        def train_epoch_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler):
            return train_epoch(train_loader, model, criterion, train_criterion_fn,
                               optimizer, compression_ctrl, epoch, config)

        # function that initializes optimizers & lr schedulers to start training
        def configure_optimizers_fn():
            params_to_optimize = get_parameter_groups(model, config)
            optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)
            return optimizer, lr_scheduler

        # register all these training-loop related funcs in nncf config
        nncf_config = register_training_loop_args(nncf_config, train_epoch_fn, validate_fn,
                                                  configure_optimizers_fn,
                                                  tensorboard_writer=config.tb,
                                                  log_dir=config.log_dir)

        # run accuracy-aware training loop
        training_runner = PTAccuracyAwareTrainingRunner(nncf_config)
        model = run_accuracy_aware_compressed_training(model, compression_ctrl, training_runner)


if __name__ == '__main__':
    main(sys.argv[1:])
