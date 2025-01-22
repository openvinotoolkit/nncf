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

import os
from copy import deepcopy
from functools import partial
from typing import Callable

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD

from nncf import NNCFConfig
from nncf.common.accuracy_aware_training.training_loop import AdaptiveCompressionTrainingLoop
from nncf.common.accuracy_aware_training.training_loop import EarlyExitCompressionTrainingLoop
from nncf.torch.initialization import register_default_init_args
from tests.torch.helpers import LeNet
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import set_torch_seed
from tests.torch.pruning.helpers import get_pruning_baseline_config
from tests.torch.quantization.quantization_helpers import get_quantization_config_without_range_init
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config


@pytest.fixture(scope="module", name="finetuned_master_lenet")
def fixture_finetuned_master_lenet():
    learning_rate = 1e-3
    finetuning_steps = 10
    with set_torch_seed():
        config = NNCFConfig({"input_info": {"sample_size": [1, *LeNet.INPUT_SIZE]}})
        train_loader = create_ones_mock_dataloader(config, num_samples=10)
        model = LeNet()
        for param in model.parameters():
            nn.init.uniform_(param, a=0.0, b=0.01)

        data_loader = iter(train_loader)
        optimizer = SGD(model.parameters(), lr=learning_rate)
        for _ in range(finetuning_steps):
            optimizer.zero_grad()
            x, y_gt = next(data_loader)
            y = model(x)
            loss = F.mse_loss(y.sum(), y_gt)
            loss.backward()
            optimizer.step()
    return model, train_loader


@pytest.fixture(name="finetuned_lenet")
def fixture_finetuned_lenet(finetuned_master_lenet):
    model, loader = finetuned_master_lenet
    return deepcopy(model), loader


def _compress_lenet_for_aa(
    model: LeNet, config: NNCFConfig, eval_fn: Callable, train_loader: torch.utils.data.DataLoader
):
    config = register_default_init_args(
        config, train_loader=train_loader, model_eval_fn=partial(eval_fn, train_loader=train_loader)
    )
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return model, compression_ctrl


# precomputed for these cases so as not to run eval in every case
LENET_UNCOMPRESSED_MODEL_ACCURACY = 0.9956666566431522


class LenetValidateFunctor:
    def __init__(self, num_steps: int):
        self._num_steps = num_steps

    def __call__(self, model, epoch=0, train_loader=None):
        with set_torch_seed():
            train_loader = iter(train_loader)
            loss = torch.FloatTensor([0])
            with torch.no_grad():
                for _ in range(self._num_steps):
                    x, y_gt = next(train_loader)
                    y = model(x)
                    loss += F.mse_loss(y.sum(), y_gt)
        return max(1 - loss.item(), 0.0)


class LeNetTrainFunctor:
    def __init__(self, train_loader: torch.utils.data.DataLoader, num_steps: int):
        self._num_steps = num_steps
        self._train_loader = train_loader

    def __call__(self, compression_ctrl, model, optimizer, **kwargs):
        with set_torch_seed():
            train_loader = iter(self._train_loader)
            for _ in range(self._num_steps):
                compression_ctrl.scheduler.step()
                optimizer.zero_grad()
                x, y_gt = next(train_loader)
                y = model(x)
                loss = F.mse_loss(y.sum(), y_gt)
                loss.backward()
                optimizer.step()


@pytest.mark.parametrize(
    ("max_accuracy_degradation", "final_compression_rate", "reference_final_metric"),
    (
        ({"maximal_relative_accuracy_degradation": 0.01}, 0.745, 0.998181),
        ({"maximal_relative_accuracy_degradation": 100.0}, 0.92, 0.0),
        ({"maximal_absolute_accuracy_degradation": 0.10}, 0.82, 0.9151),
    ),
)
def test_adaptive_compression_training_loop(
    finetuned_lenet,
    max_accuracy_degradation,
    final_compression_rate,
    reference_final_metric,
    num_steps=10,
    learning_rate=1e-3,
    initial_training_phase_epochs=5,
    patience_epochs=3,
):
    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_basic_magnitude_sparsity_config(input_sample_size=input_sample_size)

    params = {
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
        "lr_reduction_factor": 1,
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "adaptive_compression_level", "params": params}}

    validate_functor = LenetValidateFunctor(num_steps)
    config.update(accuracy_aware_config)
    original_model, train_loader = finetuned_lenet
    model, compression_ctrl = _compress_lenet_for_aa(original_model, config, validate_functor, train_loader)

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    acc_aware_training_loop = AdaptiveCompressionTrainingLoop(
        config, compression_ctrl, uncompressed_model_accuracy=LENET_UNCOMPRESSED_MODEL_ACCURACY
    )
    train_functor = LeNetTrainFunctor(train_loader, num_steps)
    model = acc_aware_training_loop.run(
        model,
        train_epoch_fn=train_functor,
        validate_fn=partial(validate_functor, train_loader=train_loader),
        configure_optimizers_fn=configure_optimizers_fn,
    )
    statistics = acc_aware_training_loop.statistics
    assert statistics.compression_rate == pytest.approx(final_compression_rate, 1e-3)
    assert statistics.compressed_accuracy == pytest.approx(reference_final_metric, 1e-4)


@pytest.mark.parametrize(
    ("max_accuracy_degradation", "maximal_total_epochs", "initial_compression_rate_step"),
    (({"maximal_absolute_accuracy_degradation": 0.1}, 1, 0.01),),
)
def test_adaptive_compression_training_loop_with_no_training(
    max_accuracy_degradation,
    maximal_total_epochs,
    initial_compression_rate_step,
    learning_rate=1e-3,
    initial_training_phase_epochs=1,
    patience_epochs=3,
):
    """When conditions below for adaptive compression training is not satisfied in the loop.
    - self.runner.compression_rate_step >= self.runner.minimal_compression_rate_step
    - self.runner.cumulative_epoch_count < self.runner.maximal_total_epochs
    """
    original_metric = 0.85

    def mock_validate_fn(model, init_step=False, epoch=0):
        if init_step:
            return original_metric

        return original_metric - 0.04 * epoch

    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_basic_magnitude_sparsity_config(input_sample_size=input_sample_size)

    params = {
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
        "maximal_total_epochs": maximal_total_epochs,
        "initial_compression_rate_step": initial_compression_rate_step,
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "adaptive_compression_level", "params": params}}

    config.update(accuracy_aware_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    config = register_default_init_args(
        config, train_loader=train_loader, model_eval_fn=partial(mock_validate_fn, init_step=True)
    )

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    def train_fn(compression_ctrl, model, optimizer, train_loader=train_loader, **kwargs):
        pass

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    acc_aware_training_loop = AdaptiveCompressionTrainingLoop(
        config, compression_ctrl, uncompressed_model_accuracy=original_metric
    )

    model = acc_aware_training_loop.run(
        model,
        train_epoch_fn=train_fn,
        validate_fn=partial(mock_validate_fn, init_step=False),
        configure_optimizers_fn=configure_optimizers_fn,
    )
    assert len(acc_aware_training_loop.runner._best_checkpoints) == 0

    possible_checkpoint_compression_rates = (
        acc_aware_training_loop.runner.get_compression_rates_with_positive_acc_budget()
    )
    assert len(possible_checkpoint_compression_rates) == 2


@pytest.mark.parametrize(
    ("max_accuracy_degradation", "maximal_total_epochs", "pruning_target"),
    (
        ({"maximal_absolute_accuracy_degradation": 0.1}, 4, 0.1),
        ({"maximal_absolute_accuracy_degradation": 0.1}, 4, 0.2),
        ({"maximal_absolute_accuracy_degradation": 0.1}, 10, 0.3),
        ({"maximal_relative_accuracy_degradation": 5.0}, 4, 0.1),
        ({"maximal_relative_accuracy_degradation": 5.0}, 4, 0.3),
    ),
)
def test_adaptive_compression_training_loop_failing(
    max_accuracy_degradation,
    maximal_total_epochs,
    pruning_target,
    initial_compression_rate_step=0.1,
    learning_rate=1e-3,
    initial_training_phase_epochs=1,
    patience_epochs=3,
    pruning_init=0.05,
    pruning_steps=1,
):
    def mock_validate_fn(model, init_step=False, epoch=0):
        original_metric = 0.85
        if init_step:
            return original_metric

        return original_metric - 0.06 * (epoch + 1)

    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_pruning_baseline_config(input_sample_size=input_sample_size)

    params = {
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
        "maximal_total_epochs": maximal_total_epochs,
        "initial_compression_rate_step": initial_compression_rate_step,
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "adaptive_compression_level", "params": params}}
    pruning_config = {
        "compression": {
            "algorithm": "filter_pruning",
            "pruning_init": pruning_init,
            "params": {"pruning_target": pruning_target, "pruning_steps": pruning_steps},
        }
    }

    config.update(accuracy_aware_config)
    config.update(pruning_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    config = register_default_init_args(
        config, train_loader=train_loader, model_eval_fn=partial(mock_validate_fn, init_step=True)
    )

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    def train_fn(compression_ctrl, model, optimizer, train_loader=train_loader, **kwargs):
        pass

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    acc_aware_training_loop = AdaptiveCompressionTrainingLoop(
        config, compression_ctrl, uncompressed_model_accuracy=LENET_UNCOMPRESSED_MODEL_ACCURACY
    )

    model = acc_aware_training_loop.run(
        model,
        train_epoch_fn=train_fn,
        validate_fn=partial(mock_validate_fn, init_step=False),
        configure_optimizers_fn=configure_optimizers_fn,
    )
    statistics = acc_aware_training_loop.statistics
    accuracy_degradation_key = next(iter(max_accuracy_degradation.keys()))
    assert (
        getattr(statistics, accuracy_degradation_key.replace("maximal_", ""))
        > config["accuracy_aware_training"]["params"][accuracy_degradation_key]
    )
    assert statistics.compression_rate < pruning_target


@pytest.mark.parametrize(
    (
        "pruning_init",
        "pruning_target",
    ),
    ((0.05, 0.1), (0.1, 0.2), (0.4, 0.6)),
)
def test_adaptive_compression_training_loop_too_high_pruning_flops(
    pruning_init,
    pruning_target,
    learning_rate=1e-3,
    maximal_relative_accuracy_degradation=1,
    initial_training_phase_epochs=1,
    patience_epochs=1,
    maximal_total_epochs=10,
    pruning_steps=1,
):
    """
    Test that ACTL reaches the maximal possible compression rate and doesn't break
    """

    mock_uncompressed_model_accuracy = 0.85

    def mock_validate_fn(model, epoch=0):
        return mock_uncompressed_model_accuracy

    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_pruning_baseline_config(input_sample_size=input_sample_size)

    params = {
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
        "maximal_total_epochs": maximal_total_epochs,
        "maximal_relative_accuracy_degradation": maximal_relative_accuracy_degradation,
    }
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "adaptive_compression_level", "params": params}}
    pruning_config = {
        "compression": {
            "algorithm": "filter_pruning",
            "pruning_init": pruning_init,
            "params": {"pruning_flops_target": pruning_target, "pruning_steps": pruning_steps},
            "ignored_scopes": ["LeNet/NNCFLinear[fc1]/linear_0", "LeNet/NNCFLinear[fc2]/linear_0"],
        }
    }

    config.update(accuracy_aware_config)
    config.update(pruning_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    config = register_default_init_args(config, train_loader=train_loader, model_eval_fn=mock_validate_fn)

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    def train_fn(compression_ctrl, model, optimizer, train_loader=train_loader, **kwargs):
        pass

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    acc_aware_training_loop = AdaptiveCompressionTrainingLoop(
        config, compression_ctrl, uncompressed_model_accuracy=mock_uncompressed_model_accuracy
    )

    model = acc_aware_training_loop.run(
        model, train_epoch_fn=train_fn, validate_fn=mock_validate_fn, configure_optimizers_fn=configure_optimizers_fn
    )

    assert acc_aware_training_loop.runner.compression_rate_target == compression_ctrl.maximal_compression_rate


@pytest.mark.parametrize(
    "max_accuracy_degradation",
    (
        ({"maximal_relative_accuracy_degradation": 30.0}),
        ({"maximal_relative_accuracy_degradation": 1.0}),
        ({"maximal_absolute_accuracy_degradation": 0.30}),
        ({"maximal_absolute_accuracy_degradation": 0.05}),
    ),
)
def test_early_exit_training_loop(
    finetuned_lenet, max_accuracy_degradation, num_steps=10, learning_rate=1e-3, maximal_total_epochs=100
):
    validate_functor = LenetValidateFunctor(num_steps)

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    params = {
        "maximal_total_epochs": maximal_total_epochs,
    }
    params.update(max_accuracy_degradation)

    accuracy_aware_config = {"accuracy_aware_training": {"mode": "early_exit", "params": params}}

    config.update(accuracy_aware_config)

    original_model, train_loader = finetuned_lenet
    model, compression_ctrl = _compress_lenet_for_aa(original_model, config, validate_functor, train_loader)

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(
        config, compression_ctrl, LENET_UNCOMPRESSED_MODEL_ACCURACY
    )
    train_functor = LeNetTrainFunctor(train_loader, num_steps)
    model = early_stopping_training_loop.run(
        model,
        train_epoch_fn=train_functor,
        validate_fn=partial(validate_functor, train_loader=train_loader),
        configure_optimizers_fn=configure_optimizers_fn,
    )
    uncompressed_model_accuracy = LENET_UNCOMPRESSED_MODEL_ACCURACY
    compressed_model_accuracy = early_stopping_training_loop.statistics.compressed_accuracy
    if "maximal_absolute_accuracy_degradation" in max_accuracy_degradation:
        assert (uncompressed_model_accuracy - compressed_model_accuracy) <= max_accuracy_degradation[
            "maximal_absolute_accuracy_degradation"
        ]
    else:
        assert (
            uncompressed_model_accuracy - compressed_model_accuracy
        ) / uncompressed_model_accuracy * 100 <= max_accuracy_degradation["maximal_relative_accuracy_degradation"]


@pytest.mark.parametrize(
    ("max_accuracy_degradation", "exit_epoch_number"),
    (
        ({"maximal_relative_accuracy_degradation": 1.0}, 6),
        ({"maximal_relative_accuracy_degradation": 30.0}, 10),
        ({"maximal_absolute_accuracy_degradation": 0.1}, 3),
    ),
)
def test_early_exit_with_mock_validation(max_accuracy_degradation, exit_epoch_number, maximal_total_epochs=100):
    original_metric = 0.85

    class mock_validate_functor:
        def __init__(self):
            self.epoch_count = 0

        def __call__(self, model, init_step=False, epoch=0):
            if init_step:
                return original_metric

            self.epoch_count = epoch
            if "maximal_relative_accuracy_degradation" in max_accuracy_degradation:
                return (
                    original_metric
                    * (1 - 0.01 * max_accuracy_degradation["maximal_relative_accuracy_degradation"])
                    * (epoch / exit_epoch_number)
                )
            return (
                (original_metric - max_accuracy_degradation["maximal_absolute_accuracy_degradation"])
                * epoch
                / exit_epoch_number
            )

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])

    params = {"maximal_total_epochs": maximal_total_epochs}
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "early_exit", "params": params}}

    config.update(accuracy_aware_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    functor_init = mock_validate_functor()
    config = register_default_init_args(
        config, train_loader=train_loader, model_eval_fn=partial(functor_init, init_step=True)
    )

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler, train_loader=train_loader):
        pass

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer, None

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(
        config, compression_ctrl, uncompressed_model_accuracy=original_metric, dump_checkpoints=False
    )

    functor_loop = mock_validate_functor()
    model = early_stopping_training_loop.run(
        model, train_epoch_fn=train_fn, validate_fn=functor_loop, configure_optimizers_fn=configure_optimizers_fn
    )
    # Epoch number starts from 0
    assert functor_loop.epoch_count == exit_epoch_number


@pytest.mark.parametrize(("max_accuracy_degradation"), (({"maximal_absolute_accuracy_degradation": 0.1}),))
def test_early_exit_with_mock_validation_and_no_improvement(max_accuracy_degradation, maximal_total_epochs=5):
    original_metric = 0.85

    def mock_validate_fn(model, init_step=False, epoch=0):
        if init_step:
            return original_metric

        return original_metric - 0.11 * (epoch + 1)

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])

    params = {"maximal_total_epochs": maximal_total_epochs}
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {"accuracy_aware_training": {"mode": "early_exit", "params": params}}

    config.update(accuracy_aware_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    config = register_default_init_args(
        config, train_loader=train_loader, model_eval_fn=partial(mock_validate_fn, init_step=True)
    )

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler, train_loader=train_loader):
        pass

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer, None

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(
        config, compression_ctrl, uncompressed_model_accuracy=original_metric, dump_checkpoints=False
    )
    assert early_stopping_training_loop.runner._best_checkpoint is None

    model = early_stopping_training_loop.run(
        model,
        train_epoch_fn=train_fn,
        validate_fn=partial(mock_validate_fn, init_step=False),
        configure_optimizers_fn=configure_optimizers_fn,
    )
    assert early_stopping_training_loop.runner._best_checkpoint is not None


@pytest.mark.parametrize(
    "aa_config",
    (
        {
            "accuracy_aware_training": {
                "mode": "early_exit",
                "params": {
                    "maximal_relative_accuracy_degradation": 1,
                    "maximal_total_epochs": 1,
                },
            },
            "compression": [
                {
                    "algorithm": "filter_pruning",
                    "pruning_init": 0.1,
                    "params": {
                        "schedule": "baseline",
                        "pruning_flops_target": 0.1,
                        "num_init_steps": 0,
                        "pruning_steps": 0,
                    },
                },
                {
                    "algorithm": "rb_sparsity",
                    "sparsity_init": 0.1,
                    "params": {"sparsity_target": 0.1, "sparsity_target_epoch": 0},
                },
            ],
        },
        {
            "accuracy_aware_training": {
                "mode": "adaptive_compression_level",
                "params": {
                    "maximal_relative_accuracy_degradation": 1,
                    "initial_training_phase_epochs": 1,
                    "maximal_total_epochs": 1,
                    "patience_epochs": 10,
                },
            },
            "compression": [
                {
                    "algorithm": "filter_pruning",
                    "pruning_init": 0.1,
                    "params": {
                        "schedule": "baseline",
                        "pruning_flops_target": 0.1,
                        "num_init_steps": 0,
                        "pruning_steps": 0,
                    },
                }
            ],
        },
    ),
)
def test_mock_dump_checkpoint(aa_config, tmp_path):
    is_called_dump_checkpoint_fn = False

    def mock_dump_checkpoint_fn(model, compression_controller, accuracy_aware_runner, aa_log_dir):
        from nncf.api.compression import CompressionAlgorithmController
        from nncf.common.accuracy_aware_training.runner import TrainingRunner

        assert isinstance(model, torch.nn.Module)
        assert isinstance(compression_controller, CompressionAlgorithmController)
        assert isinstance(accuracy_aware_runner, TrainingRunner)
        assert isinstance(aa_log_dir, str)
        nonlocal is_called_dump_checkpoint_fn
        is_called_dump_checkpoint_fn = True

        checkpoint = {
            "epoch": accuracy_aware_runner.cumulative_epoch_count + 1,
            "state_dict": model.state_dict(),
            "compression_state": compression_controller.get_compression_state(),
            "best_metric_val": accuracy_aware_runner.best_val_metric_value,
            "current_val_metric_value": accuracy_aware_runner.current_val_metric_value,
            "optimizer": accuracy_aware_runner.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(aa_log_dir, "acc_aware_checkpoint_last.pth")
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()
    config.update(aa_config)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler, train_loader=train_loader):
        pass

    def mock_validate_fn(model, init_step=False, epoch=0):
        return 80

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=0.001)
        return optimizer, None

    config = register_default_init_args(
        config, train_loader=train_loader, model_eval_fn=partial(mock_validate_fn, init_step=True)
    )

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(
        config, compression_ctrl, LENET_UNCOMPRESSED_MODEL_ACCURACY, dump_checkpoints=True
    )
    model = early_stopping_training_loop.run(
        model,
        train_epoch_fn=train_fn,
        validate_fn=partial(mock_validate_fn),
        configure_optimizers_fn=configure_optimizers_fn,
        dump_checkpoint_fn=mock_dump_checkpoint_fn,
        log_dir=tmp_path,
    )
    assert is_called_dump_checkpoint_fn
