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

from typing import Tuple

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader

from nncf.api.compression import CompressionAlgorithmController
from nncf.config import NNCFConfig
from nncf.torch.accuracy_aware_training.runner import PTAccuracyAwareTrainingRunner
from tests.torch.helpers import LeNet
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import create_random_mock_dataloader
from tests.torch.helpers import set_torch_seed
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config


def create_initialized_lenet_model_and_dataloader(
    config: NNCFConfig,
) -> Tuple[nn.Module, DataLoader, CompressionAlgorithmController]:
    with set_torch_seed():
        train_loader = create_random_mock_dataloader(config, num_samples=10)
        model = LeNet()
        for param in model.parameters():
            nn.init.uniform_(param, a=0.0, b=0.01)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    return model, train_loader, compression_ctrl


@pytest.mark.parametrize(("num_steps", "learning_rate", "reference_metric"), ((10, 5e-4, 0.78276),))
def test_runner(num_steps, learning_rate, reference_metric):
    runner = PTAccuracyAwareTrainingRunner(
        accuracy_aware_training_params={}, uncompressed_model_accuracy=0.0, dump_checkpoints=False
    )
    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_basic_magnitude_sparsity_config(input_sample_size=input_sample_size)
    model, train_loader, compression_ctrl = create_initialized_lenet_model_and_dataloader(config)

    def train_fn(compression_ctrl, model, epoch, optimizer, lr_scheduler, train_loader=train_loader):
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

    def validate_fn(model, epoch, train_loader=train_loader):
        with set_torch_seed():
            train_loader = iter(train_loader)
            loss = 0
            with torch.no_grad():
                for _ in range(num_steps):
                    x, y_gt = next(train_loader)
                    y = model(x)
                    loss += F.mse_loss(y.sum(), y_gt)
        return loss.item()

    def configure_optimizers_fn():
        optimizer = SGD(model.parameters(), lr=learning_rate)
        return optimizer, None

    runner.initialize_training_loop_fns(train_fn, validate_fn, configure_optimizers_fn, None)
    runner.initialize_logging()
    runner.reset_training()
    runner.train_epoch(model, compression_ctrl)
    metric_value = runner.validate(model)
    assert metric_value == pytest.approx(reference_metric, 1e-3)
