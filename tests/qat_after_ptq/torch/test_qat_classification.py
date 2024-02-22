# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.utils.data
import torch.utils.data.distributed
from helpers import broadcast_initialized_parameters
from helpers import get_advanced_ptq_parameters
from helpers import get_mocked_compression_ctrl
from helpers import get_num_samples
from helpers import get_quantization_preset
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import nncf
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.classification.main import create_data_loaders
from examples.torch.classification.main import create_datasets
from examples.torch.classification.main import get_argument_parser
from examples.torch.classification.main import inception_criterion_fn
from examples.torch.classification.main import train_epoch
from examples.torch.classification.main import validate
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.execution import start_worker
from examples.torch.common.model_loader import load_model
from examples.torch.common.optimizer import get_parameter_groups
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import is_pretrained_model_requested
from nncf import NNCFConfig
from nncf.torch.initialization import default_criterion_fn
from nncf.torch.utils import is_main_process
from tests.shared.paths import PROJECT_ROOT

CONFIGS = list((PROJECT_ROOT / Path("examples/torch/classification/configs/quantization")).glob("*"))


@pytest.fixture(name="quantization_config_path", params=CONFIGS, ids=[conf.stem for conf in CONFIGS])
def fixture_quantization_config(request):
    return request.param


def get_sample_config(quantization_config_path: Path, data_dir: str) -> SampleConfig:
    parser = get_argument_parser()
    data_dir = data_dir / "imagenet"
    args = parser.parse_args(["-c", str(quantization_config_path), "--data", str(data_dir), "--dataset", "imagenet"])
    sample_config = create_sample_config(args, parser)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    sample_config.device = device
    sample_config.execution_mode = get_execution_mode(sample_config)
    return sample_config


@dataclass
class DatasetSet:
    train_data_loader: torch.utils.data.DataLoader
    val_data_loader: torch.utils.data.DataLoader
    train_sampler: torch.utils.data.SequentialSampler
    calibration_dataset: nncf.Dataset


def get_datasets(sample_config: SampleConfig) -> DatasetSet:
    train_dataset, val_dataset = create_datasets(sample_config)
    train_data_lodaer, train_sampler, val_data_loader, _ = create_data_loaders(
        sample_config, train_dataset, val_dataset
    )

    def transform_fn(data_item):
        return data_item[0].to(sample_config.device)

    val_data_loader_batch_one = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    calibration_dataset = nncf.Dataset(val_data_loader_batch_one, transform_fn)
    return DatasetSet(
        train_data_loader=train_data_lodaer,
        val_data_loader=val_data_loader,
        train_sampler=train_sampler,
        calibration_dataset=calibration_dataset,
    )


def test_compression_training(quantization_config_path: Path, data_dir: str, mocker):
    if "imagenet" not in quantization_config_path.stem:
        pytest.skip("Test works only with imagenet models by far.")

    nncf_config = NNCFConfig.from_json(quantization_config_path)
    if "compression" not in nncf_config or nncf_config["compression"]["algorithm"] != "quantization":
        pytest.skip("Config without compression")

    config = get_sample_config(quantization_config_path, data_dir)
    if "accuracy_aware_training" in config:
        pytest.skip("Accuracy Aware training is not supported yet for QAT with PTQ.")

    if "pretrained" not in config or not config["pretrained"]:
        pytest.skip("Test supports only pretrained models.")

    start_worker(main_worker, config)


def main_worker(current_gpu: int, config: SampleConfig):
    configure_device(current_gpu, config)
    if is_main_process():
        configure_logging(logger, config)
    else:
        config.tb = None

    pretrained = is_pretrained_model_requested(config)
    model_name = config["model"]
    # create model
    logger.info(f"\nCreating model from config: {config.config}")
    model = load_model(
        model_name,
        pretrained=pretrained,
        num_classes=config.get("num_classes", 1000),
        model_params=config.get("model_params"),
        weights_path=config.get("weights"),
    )
    model.to(config.device)

    datasets = get_datasets(config)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(config.device)

    logger.info("Original model validation:")
    original_accuracy, *_ = validate(datasets.val_data_loader, model, criterion, config)

    logger.info("Apply quantization to the model:")
    config_quantization_params = config["compression"]

    preset = get_quantization_preset(config_quantization_params)
    advanced_parameters = get_advanced_ptq_parameters(config_quantization_params)
    subset_size = get_num_samples(config_quantization_params)

    quantized_model = nncf.quantize(
        model,
        datasets.calibration_dataset,
        preset=preset,
        advanced_parameters=advanced_parameters,
        subset_size=subset_size,
    )

    acc_drop = train(
        quantized_model,
        config,
        criterion,
        datasets,
        original_accuracy,
        get_mocked_compression_ctrl(),
    )
    assert accuracy_drop_is_acceptable(acc_drop)


def accuracy_drop_is_acceptable(acc_drop: float) -> bool:
    """
    Returns True in case acc_drop is less than 1 precent.
    """
    return acc_drop < 1.0


def catch_nans_wrapper(fn):
    def wrapper(*args, **kwargs):
        with torch.autograd.set_detect_anomaly(True):
            fn(*args, **kwargs)

    return wrapper


def train(
    model: torch.nn.Module,
    config: SampleConfig,
    criterion: torch.nn.Module,
    datasets: DatasetSet,
    original_accuracy: float,
    compression_ctrl,
) -> float:
    """
    :return: Accuracy drop between original accuracy and trained quantized model accuracy.
    """
    model, _ = prepare_model_for_execution(model, config)
    if config.distributed:
        broadcast_initialized_parameters(model)
    model_name = config["model"]
    train_criterion_fn = inception_criterion_fn if "inception" in model_name else default_criterion_fn
    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)
    best_acc1 = 0
    logger.info("Qantization aware training pipeline starts.")
    for epoch in range(config.start_epoch, config.epochs + 1):
        current_accuracy, *_ = validate(datasets.val_data_loader, model, criterion, config, epoch - 1)
        best_acc1 = max(current_accuracy, best_acc1)
        acc_drop = original_accuracy - current_accuracy
        logger.info(f"Metric: {current_accuracy}, FP32 diff: {acc_drop}")
        if accuracy_drop_is_acceptable(acc_drop):
            logger.info(f"Accuracy is within 1 percent drop," f" pipeline is making early exit on epoch {epoch - 1}")
            logger.info(
                f"Epochs in config: {config.epochs}, epochs trained: {epoch}, epochs saved: {config.epochs - epoch}"
            )
            return acc_drop
        if epoch == config.epochs:
            logger.info("Training pipeline is finished, accuracy was not recovered.")
            return acc_drop

        # update compression scheduler state at the begin of the epoch
        if config.distributed:
            datasets.train_sampler.set_epoch(epoch)

        # train for one epoch
        train_epoch(
            datasets.train_data_loader, model, criterion, train_criterion_fn, optimizer, compression_ctrl, epoch, config
        )

        # Learning rate scheduling should be applied after optimizer’s update
        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_acc1)
