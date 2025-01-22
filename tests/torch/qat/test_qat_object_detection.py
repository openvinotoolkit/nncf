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

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
import torch
import torch.utils.data
import torch.utils.data.distributed
from helpers import broadcast_initialized_parameters
from helpers import get_advanced_ptq_parameters
from helpers import get_mocked_compression_ctrl
from helpers import get_num_samples
from helpers import get_quantization_preset
from helpers import start_worker_clean_memory
from torch.optim.lr_scheduler import ReduceLROnPlateau

import nncf
from examples.common.sample_config import SampleConfig
from examples.common.sample_config import create_sample_config
from examples.torch.common.example_logger import logger
from examples.torch.common.execution import get_execution_mode
from examples.torch.common.execution import prepare_model_for_execution
from examples.torch.common.optimizer import get_parameter_groups
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import is_on_first_rank
from examples.torch.object_detection.dataset import detection_collate
from examples.torch.object_detection.dataset import get_testing_dataset
from examples.torch.object_detection.eval import test_net as sample_validate
from examples.torch.object_detection.layers.modules import MultiBoxLoss
from examples.torch.object_detection.main import create_dataloaders
from examples.torch.object_detection.main import create_model
from examples.torch.object_detection.main import get_argument_parser
from examples.torch.object_detection.main import train_epoch
from nncf import NNCFConfig
from nncf.common.compression import BaseCompressionAlgorithmController
from tests.cross_fw.shared.paths import PROJECT_ROOT

CONFIGS = list((PROJECT_ROOT / Path("examples/torch/object_detection/configs")).glob("*"))


def _get_filtered_quantization_configs() -> List[Path]:
    configs = []
    for quantization_config_path in CONFIGS:
        nncf_config = NNCFConfig.from_json(quantization_config_path)
        if (
            "compression" not in nncf_config
            or isinstance(nncf_config["compression"], list)
            or nncf_config["compression"]["algorithm"] != "quantization"
        ):
            # Config without compression
            continue

        if "accuracy_aware_training" in nncf_config:
            # Accuracy Aware training is not supported yet for QAT with PTQ.
            continue

        configs.append(quantization_config_path)
    return configs


FILTERED_CONFIGS = _get_filtered_quantization_configs()


@pytest.fixture(name="quantization_config_path", params=FILTERED_CONFIGS, ids=[conf.stem for conf in FILTERED_CONFIGS])
def fixture_quantization_config_path(request):
    return request.param


def get_sample_config(quantization_config_path: Path, data_dir: Path, weights_dir: Path) -> SampleConfig:
    parser = get_argument_parser()
    weights_path = weights_dir / "object_detection" / "voc" / (quantization_config_path.stem.split("_int8")[0] + ".pth")
    args = parser.parse_args(
        [
            "-c",
            str(quantization_config_path),
            "--data",
            str(data_dir),
            "--dataset",
            "voc",
            "--weights",
            str(weights_path),
        ]
    )
    sample_config = create_sample_config(args, parser)
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    sample_config.device = device
    sample_config.execution_mode = get_execution_mode(sample_config)

    if sample_config.dataset_dir is not None:
        sample_config.train_imgs = sample_config.train_anno = sample_config.test_imgs = sample_config.test_anno = (
            sample_config.dataset_dir
        )
    return sample_config


@dataclass
class DatasetSet:
    train_data_loader: torch.utils.data.DataLoader
    test_data_loader: torch.utils.data.DataLoader
    calibration_dataset: nncf.Dataset


def get_datasets(config: SampleConfig) -> DatasetSet:
    test_data_loader, train_data_loader, _ = create_dataloaders(config)

    test_dataset = get_testing_dataset(config.dataset, config.test_anno, config.test_imgs, config)
    logger.info("Loaded {} testing images".format(len(test_dataset)))
    if config.distributed:
        test_sampler = torch.utils.data.DistributedSampler(test_dataset, config.rank, config.world_size)
    else:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    def transform_fn(data_item):
        return data_item[0].to(config.device)

    val_data_loader_batch_one = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=config.workers,
        shuffle=False,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=False,
        sampler=test_sampler,
    )

    calibration_dataset = nncf.Dataset(val_data_loader_batch_one, transform_fn)
    return DatasetSet(
        train_data_loader=train_data_loader,
        test_data_loader=test_data_loader,
        calibration_dataset=calibration_dataset,
    )


def accuracy_drop_is_acceptable(acc_drop: float) -> bool:
    """
    Returns True in case acc_drop is less than 1 percent.
    """
    return acc_drop < 0.01


def validate(net: torch.nn.Module, device, data_loader, distributed):
    with torch.no_grad():
        net.eval()
        return sample_validate(net, device, data_loader, distributed)


def get_optimizer_and_lr_scheduler(config: SampleConfig, model: torch.nn.Module):
    params_to_optimize = get_parameter_groups(model, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)
    return optimizer, lr_scheduler


def train(
    model: torch.nn.Module,
    config: SampleConfig,
    criterion: torch.nn.Module,
    datasets: DatasetSet,
    original_metric: float,
    compression_ctrl: BaseCompressionAlgorithmController,
) -> float:
    """
    :return: Accuracy drop between original accuracy and trained quantized model accuracy.
    """
    model, _ = prepare_model_for_execution(model, config)
    if config.distributed:
        broadcast_initialized_parameters(model)

    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(config, model)

    best_metric = 0
    loc_loss = 0
    conf_loss = 0

    epoch_size = len(datasets.train_data_loader)
    logger.info("Quantization aware training pipeline starts.")
    for epoch in range(config.start_epoch, config.epochs + 1):
        current_metric = validate(
            model, config.device, datasets.test_data_loader, distributed=config.multiprocessing_distributed
        )
        best_metric = max(current_metric, best_metric)
        acc_drop = original_metric - current_metric
        logger.info(f"Metric: {current_metric}, FP32 diff: {acc_drop}")
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
        model.train()
        train_epoch(
            compression_ctrl,
            model,
            config,
            datasets.train_data_loader,
            criterion,
            optimizer,
            epoch_size,
            epoch,
            loc_loss,
            conf_loss,
        )

        # Learning rate scheduling should be applied after optimizer’s update
        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_metric)


def check_training_correctness(
    config: SampleConfig, model: torch.nn.Module, datasets: DatasetSet, criterion: torch.nn.Module
):
    """
    This function tries to run 50 training steps for one input and target pair and
    checks loss decreases. This is needed to check model with compression could be
    trained after the PTQ.
    """
    logger.info("Check model is trainable...")
    steps_to_check = 50
    optimizer, _ = get_optimizer_and_lr_scheduler(config, model)
    images, targets, *_ = next(iter(datasets.calibration_dataset.get_data()))
    images = images.to(config.device)
    targets = [t.to(config.device) for t in targets]
    with torch.no_grad():
        images = torch.cat([images, images], dim=0)
        targets.append(targets[0])
    loss_list = []
    model.train()
    for _ in range(steps_to_check):
        output = model(images)
        loss_l, loss_c = criterion(output, targets)
        loss = loss_l + loss_c
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss_list[-1] < loss_list[0]


def main_worker(current_gpu: int, config: SampleConfig):
    configure_device(current_gpu, config)
    if is_on_first_rank(config):
        configure_logging(logger, config)

    # create model
    logger.info(f"\nCreating model from config: {config.config}")
    model = create_model(config)

    datasets = get_datasets(config)
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
    criterion = criterion.to(config.device)

    logger.info("Original model validation:")
    original_metric = validate(model, config.device, datasets.test_data_loader, config.distributed)

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
    if config.distributed:
        config.batch_size //= config.ngpus_per_node
        config.workers //= config.ngpus_per_node

    acc_drop = train(quantized_model, config, criterion, datasets, original_metric, get_mocked_compression_ctrl())
    assert accuracy_drop_is_acceptable(acc_drop)
    check_training_correctness(config, model, datasets, criterion)
    logger.info("Done!")


@pytest.mark.weekly
def test_compression_training(quantization_config_path: Path, sota_data_dir, sota_checkpoints_dir):
    sample_config = get_sample_config(quantization_config_path, sota_data_dir, sota_checkpoints_dir)
    start_worker_clean_memory(main_worker, sample_config)
