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

import os
import pytest
from functools import partial

import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F

from nncf.torch import AdaptiveCompressionTrainingLoop
from nncf.torch import EarlyExitCompressionTrainingLoop
from nncf.torch.initialization import register_default_init_args

from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import LeNet
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import set_torch_seed
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.torch.quantization.test_quantization_helpers import get_quantization_config_without_range_init


def create_finetuned_lenet_model_and_dataloader(config, eval_fn, finetuning_steps,
                                                learning_rate=1e-3):
    with set_torch_seed():
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

    config = register_default_init_args(config,
                                        train_loader=train_loader,
                                        model_eval_fn=partial(eval_fn, train_loader=train_loader))
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return model, train_loader, compression_ctrl


@pytest.mark.parametrize(
    ('max_accuracy_degradation',
     'final_compression_rate',
     'reference_final_metric'),
    (
            ({'maximal_relative_accuracy_degradation': 0.01}, 0.66742, 0.996252),
            ({'maximal_relative_accuracy_degradation': 100.0}, 0.94136, 0.876409),
            ({'maximal_absolute_accuracy_degradation': 0.10}, 0.767040, 0.938572),
    )
)
def test_adaptive_compression_training_loop(max_accuracy_degradation,
                                            final_compression_rate,
                                            reference_final_metric,
                                            num_steps=10, learning_rate=1e-3,
                                            initial_training_phase_epochs=5,
                                            patience_epochs=3,
                                            init_finetuning_steps=10):
    def validate_fn(model, epoch=0, train_loader=None):
        with set_torch_seed():
            train_loader = iter(train_loader)
            loss = torch.FloatTensor([0])
            with torch.no_grad():
                for _ in range(num_steps):
                    x, y_gt = next(train_loader)
                    y = model(x)
                    loss += F.mse_loss(y.sum(), y_gt)
        return 1 - loss.item()

    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_basic_magnitude_sparsity_config(input_sample_size=input_sample_size)

    params = {
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {
        "accuracy_aware_training": {
            "mode": "adaptive_compression_level",
            "params": params
        }
    }

    config.update(accuracy_aware_config)

    model, train_loader, compression_ctrl = create_finetuned_lenet_model_and_dataloader(config,
                                                                                        validate_fn,
                                                                                        init_finetuning_steps)

    def train_fn(compression_ctrl, model, optimizer,
                 train_loader=train_loader, **kwargs):
        with set_torch_seed():
            train_loader = iter(train_loader)
            for _ in range(num_steps):
                compression_ctrl.scheduler.step()
                optimizer.zero_grad()
                x, y_gt = next(train_loader)
                y = model(x)
                loss = F.mse_loss(y.sum(), y_gt)
                loss.backward()
                optimizer.step()

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    acc_aware_training_loop = AdaptiveCompressionTrainingLoop(config, compression_ctrl)
    model = acc_aware_training_loop.run(model,
                                        train_epoch_fn=train_fn,
                                        validate_fn=partial(validate_fn, train_loader=train_loader),
                                        configure_optimizers_fn=configure_optimizers_fn)
    assert compression_ctrl.compression_rate == pytest.approx(final_compression_rate, 1e-3)
    assert validate_fn(model, train_loader=train_loader) == pytest.approx(reference_final_metric, 1e-4)


@pytest.mark.parametrize(
    'max_accuracy_degradation',
    (({'maximal_relative_accuracy_degradation': 30.0}), ({'maximal_relative_accuracy_degradation': 1.0}),
     ({'maximal_absolute_accuracy_degradation': 0.30}), ({'maximal_absolute_accuracy_degradation': 0.05}))
)
def test_early_exit_training_loop(max_accuracy_degradation,
                                  num_steps=10, learning_rate=1e-3,
                                  maximal_total_epochs=100,
                                  init_finetuning_steps=10):
    def validate_fn(model, epoch=0, train_loader=None):
        with set_torch_seed():
            train_loader = iter(train_loader)
            loss = torch.FloatTensor([0])
            with torch.no_grad():
                for _ in range(num_steps):
                    x, y_gt = next(train_loader)
                    y = model(x)
                    loss += F.mse_loss(y.sum(), y_gt)
        return 1 - loss.item()

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    params = {
        "maximal_total_epochs": maximal_total_epochs,
    }
    params.update(max_accuracy_degradation)

    accuracy_aware_config = {
        "accuracy_aware_training": {
            "mode": "early_exit",
            "params": params
        }
    }

    config.update(accuracy_aware_config)

    model, train_loader, compression_ctrl = create_finetuned_lenet_model_and_dataloader(config,
                                                                                        validate_fn,
                                                                                        init_finetuning_steps)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler,
                 train_loader=train_loader):
        with set_torch_seed():
            train_loader = iter(train_loader)
            for _ in range(num_steps):
                compression_ctrl.scheduler.step()
                optimizer.zero_grad()
                x, y_gt = next(train_loader)
                y = model(x)
                loss = F.mse_loss(y.sum(), y_gt)
                loss.backward()
                optimizer.step()

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(config, compression_ctrl)
    model = early_stopping_training_loop.run(model,
                                             train_epoch_fn=train_fn,
                                             validate_fn=partial(validate_fn, train_loader=train_loader),
                                             configure_optimizers_fn=configure_optimizers_fn)
    original_model_accuracy = model.original_model_accuracy
    compressed_model_accuracy = validate_fn(model, train_loader=train_loader)
    if "maximal_absolute_accuracy_degradation" in max_accuracy_degradation:
        assert (original_model_accuracy - compressed_model_accuracy) <= \
               max_accuracy_degradation["maximal_absolute_accuracy_degradation"]
    else:
        assert (original_model_accuracy - compressed_model_accuracy) / original_model_accuracy * 100 <= \
               max_accuracy_degradation["maximal_relative_accuracy_degradation"]


@pytest.mark.parametrize(
    ('max_accuracy_degradation', 'exit_epoch_number'),
    (({'maximal_relative_accuracy_degradation': 1.0}, 6),
     ({'maximal_relative_accuracy_degradation': 30.0}, 10),
     ({'maximal_absolute_accuracy_degradation': 0.1}, 3))
)
def test_early_exit_with_mock_validation(max_accuracy_degradation, exit_epoch_number,
                                         maximal_total_epochs=100):
    epoch_counter = 0

    def mock_validate_fn(model, init_step=False, epoch=0):
        original_metric = 0.85
        if init_step:
            return original_metric
        nonlocal epoch_counter
        epoch_counter = epoch
        if "maximal_relative_accuracy_degradation" in max_accuracy_degradation:
            return original_metric * (1 - 0.01 * max_accuracy_degradation['maximal_relative_accuracy_degradation']) * (
                    epoch / exit_epoch_number)
        return (original_metric - max_accuracy_degradation['maximal_absolute_accuracy_degradation']) * \
               epoch / exit_epoch_number

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])

    params = {
        "maximal_total_epochs": maximal_total_epochs
    }
    params.update(max_accuracy_degradation)
    accuracy_aware_config = {
        "accuracy_aware_training": {
            "mode": "early_exit",
            "params": params
        }
    }

    config.update(accuracy_aware_config)

    train_loader = create_ones_mock_dataloader(config, num_samples=10)
    model = LeNet()

    config = register_default_init_args(config,
                                        train_loader=train_loader,
                                        model_eval_fn=partial(mock_validate_fn, init_step=True))

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler,
                 train_loader=train_loader):
        pass

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=1e-3)
        return optimizer, None

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(config, compression_ctrl,
                                                                    dump_checkpoints=False)
    model = early_stopping_training_loop.run(model,
                                             train_epoch_fn=train_fn,
                                             validate_fn=partial(mock_validate_fn),
                                             configure_optimizers_fn=configure_optimizers_fn)
    # Epoch number starts from 0
    assert epoch_counter == exit_epoch_number


@pytest.mark.parametrize('aa_config', (
        {
            "accuracy_aware_training": {
                "mode": "early_exit",
                "params": {
                    "maximal_relative_accuracy_degradation": 1,
                    "maximal_total_epochs": 1,
                }
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
                    }
                },
                {
                    "algorithm": "rb_sparsity",
                    "sparsity_init": 0.1,
                    "params": {
                        "sparsity_target": 0.1,
                        "sparsity_target_epoch": 0
                    }
                }
            ]
        },
        {
            "accuracy_aware_training": {
                "mode": "adaptive_compression_level",
                "params": {
                    "maximal_relative_accuracy_degradation": 1,
                    "initial_training_phase_epochs": 1,
                    "maximal_total_epochs": 1,
                    "patience_epochs": 10
                }
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
                    }
                }
            ]
        }
)
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
            'epoch': accuracy_aware_runner.cumulative_epoch_count + 1,
            'state_dict': model.state_dict(),
            'compression_state': compression_controller.get_compression_state(),
            'best_metric_val': accuracy_aware_runner.best_val_metric_value,
            'current_val_metric_value': accuracy_aware_runner.current_val_metric_value,
            'optimizer': accuracy_aware_runner.optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(aa_log_dir, 'acc_aware_checkpoint_last.pth')
        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    config = get_quantization_config_without_range_init(LeNet.INPUT_SIZE[-1])
    train_loader = create_ones_mock_dataloader(aa_config, num_samples=10)
    model = LeNet()
    config.update(aa_config)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler,
                 train_loader=train_loader):
        pass

    def mock_validate_fn(model, init_step=False, epoch=0):
        return 80

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=0.001)
        return optimizer, None

    config = register_default_init_args(config,
                                        train_loader=train_loader,
                                        model_eval_fn=partial(mock_validate_fn, init_step=True))

    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

    early_stopping_training_loop = EarlyExitCompressionTrainingLoop(config, compression_ctrl,
                                                                    dump_checkpoints=True)
    model = early_stopping_training_loop.run(model,
                                             train_epoch_fn=train_fn,
                                             validate_fn=partial(mock_validate_fn),
                                             configure_optimizers_fn=configure_optimizers_fn,
                                             dump_checkpoint_fn=mock_dump_checkpoint_fn,
                                             log_dir=tmp_path)
    assert is_called_dump_checkpoint_fn
