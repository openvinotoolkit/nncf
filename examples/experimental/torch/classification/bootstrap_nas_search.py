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
import warnings
from pathlib import Path
from shutil import copyfile

import torch
from torch import nn

from examples.common.paths import configure_paths
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.classification.main import create_data_loaders
from examples.torch.classification.main import create_datasets
from examples.torch.classification.main import get_argument_parser
from examples.torch.classification.main import validate
from examples.torch.common.argparser import parse_args
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import set_seed
from examples.torch.common.execution import start_worker
from examples.torch.common.model_loader import load_model
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import create_code_snapshot
from examples.torch.common.utils import get_run_name
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.common.utils import print_args
from nncf.config.structures import BNAdaptationInitArgs
from nncf.experimental.torch.nas.bootstrapNAS import BaseSearchAlgorithm
from nncf.experimental.torch.nas.bootstrapNAS.training.model_creator_helpers import resume_compression_from_state
from nncf.torch.checkpoint_loading import load_state
from nncf.torch.initialization import wrap_dataloader_for_init
from nncf.torch.model_creation import create_nncf_network
from nncf.torch.utils import is_main_process


def get_nas_argument_parser():
    parser = get_argument_parser()
    parser.add_argument(
        "--train-steps", default=None, type=int, help="Enables running training for the given number of steps"
    )

    parser.add_argument("--elasticity-state-path", required=True, type=str, help="Path of elasticity state")

    parser.add_argument("--supernet-weights", required=True, type=str, help="Path to weights of trained super-network")

    parser.add_argument("--search-mode", "-s", action="store_true", help="Activates search mode")

    return parser


def main(argv):
    parser = get_nas_argument_parser()
    args = parse_args(parser, argv)
    config = create_sample_config(args, parser)
    config.search_elasticity_state_path = args.elasticity_state_path
    config.search_supernet_weights = args.supernet_weights
    config.search_mode_active = args.search_mode

    if config.dist_url == "env://":
        config.update_from_env()

    configure_paths(config, get_run_name(config))
    copyfile(args.config, osp.join(config.log_dir, "config.json"))
    source_root = Path(__file__).absolute().parents[2]  # nncf root
    create_code_snapshot(source_root, osp.join(config.log_dir, "snapshot.tar.gz"))

    if config.seed is not None:
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    config.execution_mode = get_execution_mode(config)

    start_worker(main_worker, config)


def main_worker(current_gpu, config: SampleConfig):
    configure_device(current_gpu, config)
    if is_main_process():
        configure_logging(logger, config)
        print_args(config)

    set_seed(config)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    model_name = config["model"]

    nncf_config = config.nncf_config
    pretrained = is_pretrained_model_requested(config)

    # Data loading code
    train_dataset, val_dataset = create_datasets(config)
    train_loader, _, val_loader, _ = create_data_loaders(config, train_dataset, val_dataset)

    bn_adapt_args = BNAdaptationInitArgs(data_loader=wrap_dataloader_for_init(train_loader), device=config.device)
    nncf_config.register_extra_structs([bn_adapt_args])
    # create model
    model = load_model(
        model_name,
        pretrained=pretrained,
        num_classes=config.get("num_classes", 1000),
        model_params=config.get("model_params"),
        weights_path=config.get("weights"),
    )

    model.to(config.device)

    if model_name == "efficient_net":
        model.set_swish(memory_efficient=False)

    validate(val_loader, model, criterion, config)

    def validate_model_fn(model_, loader):
        top1, top5, loss = validate(loader, model_, criterion, config, log_validation_info=False)
        return top1, top5, loss

    def validate_model_fn_top1(model_, loader_):
        top1, _, _ = validate_model_fn(model_, loader_)
        return top1

    nncf_network = create_nncf_network(model, nncf_config)

    if config.search_mode_active:
        compression_state = torch.load(config.search_elasticity_state_path)
        model, elasticity_ctrl = resume_compression_from_state(nncf_network, compression_state)
        model_weights = torch.load(config.search_supernet_weights)

        load_state(model, model_weights, is_resume=True)

        top1_acc = validate_model_fn_top1(model, val_loader)
        logger.info("SuperNetwork Top 1: {top1_acc}".format(top1_acc=top1_acc))

        search_algo = BaseSearchAlgorithm.from_config(model, elasticity_ctrl, nncf_config)

        elasticity_ctrl, best_config, performance_metrics = search_algo.run(
            validate_model_fn_top1, val_loader, config.checkpoint_save_dir, tensorboard_writer=config.tb
        )

        logger.info(f"Best config: {best_config}")
        logger.info(f"Performance metrics: {performance_metrics}")
        search_algo.visualize_search_progression()

        # Maximal subnet
        elasticity_ctrl.multi_elasticity_handler.activate_maximum_subnet()
        search_algo.bn_adaptation.run(nncf_network)
        top1_acc = validate_model_fn_top1(nncf_network, val_loader)
        logger.info(
            "Maximal subnet Top1 acc: {top1_acc}, Macs: {macs}".format(
                top1_acc=top1_acc,
                macs=elasticity_ctrl.multi_elasticity_handler.count_flops_and_weights_for_active_subnet()[0] / 2000000,
            )
        )

        # Best found subnet
        elasticity_ctrl.multi_elasticity_handler.activate_subnet_for_config(best_config)
        search_algo.bn_adaptation.run(nncf_network)
        top1_acc = validate_model_fn_top1(nncf_network, val_loader)
        logger.info(
            "Best found subnet Top1 acc: {top1_acc}, Macs: {macs}".format(
                top1_acc=top1_acc,
                macs=elasticity_ctrl.multi_elasticity_handler.count_flops_and_weights_for_active_subnet()[0] / 2000000,
            )
        )
        elasticity_ctrl.export_model(osp.join(config.log_dir, "best_subnet.onnx"))

        search_algo.search_progression_to_csv()
        search_algo.evaluators_to_csv()

        assert best_config == elasticity_ctrl.multi_elasticity_handler.get_active_config()

    if "test" in config.mode:
        validate(val_loader, nncf_network, criterion, config)


if __name__ == "__main__":
    main(sys.argv[1:])
