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
from examples.torch.common.model_loader import load_model
from examples.torch.common.optimizer import make_optimizer
from examples.torch.common.utils import configure_device
from examples.torch.common.utils import configure_logging
from examples.torch.common.utils import is_pretrained_model_requested
from examples.torch.semantic_segmentation.main import get_arguments_parser
from examples.torch.semantic_segmentation.main import get_criterion
from examples.torch.semantic_segmentation.main import get_dataset
from examples.torch.semantic_segmentation.main import get_joint_transforms
from examples.torch.semantic_segmentation.main import get_params_to_optimize
from examples.torch.semantic_segmentation.main import load_dataset
from examples.torch.semantic_segmentation.main import test as sample_validate
from examples.torch.semantic_segmentation.metric import IoU
from examples.torch.semantic_segmentation.test import Test
from examples.torch.semantic_segmentation.train import Train
from examples.torch.semantic_segmentation.utils.loss_funcs import do_model_specific_postprocessing
from nncf import NNCFConfig
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.torch.utils import is_main_process
from tests.cross_fw.shared.paths import PROJECT_ROOT

CONFIGS = list((PROJECT_ROOT / Path("examples/torch/semantic_segmentation/configs")).glob("*"))


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
    parser = get_arguments_parser()
    meta = None
    datasets_meta = [{"name": "mapillary", "dir_name": "mapillary_vistas"}, {"name": "camvid", "dir_name": "camvid"}]
    for datset_meta in datasets_meta:
        if datset_meta["name"] in quantization_config_path.stem:
            meta = datset_meta
            break
    else:
        raise RuntimeError(f"Dataset for the config {str(quantization_config_path)} is unknown.")

    weights_path = (
        weights_dir / "segmentation" / meta["name"] / (quantization_config_path.stem.split("_int8")[0] + ".pth")
    )
    data_dir = data_dir / meta["dir_name"]
    args = parser.parse_args(
        [
            "-c",
            str(quantization_config_path),
            "--data",
            str(data_dir),
            "--dataset",
            meta["name"],
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
    return sample_config


@dataclass
class DatasetSet:
    train_data_loader: torch.utils.data.DataLoader
    val_data_loader: torch.utils.data.DataLoader
    class_weights: object
    calibration_dataset: nncf.Dataset


def get_datasets(dataset, config: SampleConfig) -> DatasetSet:
    loaders, w_class = load_dataset(dataset, config)
    train_loader, val_loader, _ = loaders
    transforms_val = get_joint_transforms(is_train=False, config=config)
    # Get selected dataset
    val_dataset = dataset(config.dataset_dir, image_set="val", transforms=transforms_val)

    def transform_fn(data_item):
        return data_item[0].to(config.device)

    val_data_loader_batch_one = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
    )
    calibration_dataset = nncf.Dataset(val_data_loader_batch_one, transform_fn)
    return DatasetSet(
        train_data_loader=train_loader,
        val_data_loader=val_loader,
        class_weights=w_class,
        calibration_dataset=calibration_dataset,
    )


def accuracy_drop_is_acceptable(acc_drop: float) -> bool:
    """
    Returns True in case acc_drop is less than 1 percent.
    """
    return acc_drop < 0.01


def get_optimizer_and_lr_scheduler(config: SampleConfig, model_without_dp: torch.nn.Module):
    optim_config = config.get("optimizer", {})
    optim_params = optim_config.get("optimizer_params", {})
    lr = optim_params.get("lr", 1e-4)

    params_to_optimize = get_params_to_optimize(model_without_dp, lr * 10, config)
    optimizer, lr_scheduler = make_optimizer(params_to_optimize, config)
    return optimizer, lr_scheduler


def train(
    model: torch.nn.Module,
    model_without_dp: torch.nn.Module,
    config: SampleConfig,
    criterion: torch.nn.Module,
    datasets: DatasetSet,
    original_metric: float,
    color_encoding: object,
    compression_ctrl: BaseCompressionAlgorithmController,
) -> float:
    """
    :return: Accuracy drop between original accuracy and trained quantized model accuracy.
    """
    logger.info("\nTraining...\n")

    optimizer, lr_scheduler = get_optimizer_and_lr_scheduler(config, model_without_dp)

    # Evaluation metric

    ignore_index = None
    ignore_unlabeled = config.get("ignore_unlabeled", True)
    if ignore_unlabeled and ("unlabeled" in color_encoding):
        ignore_index = list(color_encoding).index("unlabeled")

    metric = IoU(len(color_encoding), ignore_index=ignore_index)

    best_miou = -1

    # Start Training
    train_obj = Train(
        model, datasets.train_data_loader, optimizer, criterion, compression_ctrl, metric, config.device, config.model
    )
    val_obj = Test(model, datasets.val_data_loader, criterion, metric, config.device, config.model)

    logger.info("Quantization aware training pipeline starts.")
    for epoch in range(config.start_epoch, config.epochs):
        if config.distributed:
            datasets.train_data_loader.sampler.set_epoch(epoch)

        logger.info(">>>> [Epoch: {0:d}] Validation".format(epoch))
        _, (_, current_miou) = val_obj.run_epoch(config.print_step)
        # best_metric = max(current_miou, best_metric)
        acc_drop = original_metric - current_miou
        best_miou = max(current_miou, best_miou)
        logger.info(f"Metric: {current_miou}, FP32 diff: {acc_drop}")
        if accuracy_drop_is_acceptable(acc_drop):
            logger.info(f"Accuracy is within 1 percent drop," f" pipeline is making early exit on epoch {epoch - 1}")
            logger.info(
                f"Epochs in config: {config.epochs}, epochs trained: {epoch}, epochs saved: {config.epochs - epoch}"
            )
            return acc_drop
        if epoch == config.epochs:
            logger.info("Training pipeline is finished, accuracy was not recovered.")
            return acc_drop

        logger.info(">>>> [Epoch: {0:d}] Training".format(epoch))
        epoch_loss, (_, miou) = train_obj.run_epoch(config.print_step)

        logger.info(">>>> [Epoch: {0:d}] Avg. loss: {1:.4f} | Mean IoU: {2:.4f}".format(epoch, epoch_loss, miou))

        lr_scheduler.step(epoch if not isinstance(lr_scheduler, ReduceLROnPlateau) else best_miou)


def check_training_correctness(
    config: SampleConfig,
    model: torch.nn.Module,
    datasets: DatasetSet,
    criterion: torch.nn.Module,
):
    """
    This function tries to run 50 training steps for one input and target pair and
    checks loss decreases. This is needed to check model with compression could be
    trained after the PTQ.
    """
    logger.info("Check model is trainable...")
    steps_to_check = 50
    model_without_dp = model
    if hasattr(model_without_dp, "module"):
        model_without_dp = model_without_dp.module

    optimizer, _ = get_optimizer_and_lr_scheduler(config, model_without_dp)
    input_, labels, *_ = next(iter(datasets.calibration_dataset.get_data()))
    input_ = input_.to(config.device)
    labels = labels.to(config.device)
    # Make batch_size==2 to make batchnorms work
    with torch.no_grad():
        input_ = torch.cat([input_, input_], dim=0)
        labels = torch.cat([labels, labels], dim=0)
    loss_list = []
    model.train()
    for _ in range(steps_to_check):
        outputs = model(input_)
        labels, loss_outputs, _ = do_model_specific_postprocessing(config.model, labels, outputs)

        # Loss computation
        loss = criterion(loss_outputs, labels)
        loss_list.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert loss_list[-1] < loss_list[0]


def main_worker(current_gpu: int, config: SampleConfig):
    configure_device(current_gpu, config)
    if is_main_process():
        configure_logging(logger, config)

    # create model
    logger.info(f"\nCreating model from config: {config.config}")

    dataset = get_dataset(config.dataset)
    color_encoding = dataset.color_encoding
    num_classes = len(color_encoding)

    pretrained = is_pretrained_model_requested(config)
    model = load_model(
        config.model,
        pretrained=pretrained,
        num_classes=num_classes,
        model_params=config.get("model_params", {}),
        weights_path=config.get("weights"),
    )
    model.to(config.device)

    datasets = get_datasets(dataset, config)
    criterion = get_criterion(datasets.class_weights, config)

    logger.info("Original model validation:")
    original_metric = sample_validate(model, datasets.val_data_loader, criterion, color_encoding, config)

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
    model, model_without_dp = prepare_model_for_execution(model, config)
    if config.distributed:
        broadcast_initialized_parameters(model)

    acc_drop = train(
        quantized_model,
        model_without_dp,
        config,
        criterion,
        datasets,
        original_metric,
        color_encoding,
        get_mocked_compression_ctrl(),
    )
    assert accuracy_drop_is_acceptable(acc_drop)
    check_training_correctness(config, quantized_model, datasets, criterion)
    logger.info("Done!")


@pytest.mark.weekly
def test_compression_training(quantization_config_path: Path, sota_data_dir, sota_checkpoints_dir):
    sample_config = get_sample_config(quantization_config_path, sota_data_dir, sota_checkpoints_dir)
    start_worker_clean_memory(main_worker, sample_config)
