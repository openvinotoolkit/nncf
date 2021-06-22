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

import pytest
from functools import partial

import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional as F

from nncf.torch import PTAdaptiveCompressionTrainingLoop
from nncf.torch.initialization import register_default_init_args

from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import LeNet
from tests.torch.helpers import create_ones_mock_dataloader
from tests.torch.helpers import set_torch_seed
from tests.torch.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config


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
        (0.01, 0.66742, 0.996252),
        (100., 0.94136, 0.876409),
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
            loss = 0
            with torch.no_grad():
                for _ in range(num_steps):
                    x, y_gt = next(train_loader)
                    y = model(x)
                    loss += F.mse_loss(y.sum(), y_gt)
        return 1-loss.item()

    input_sample_size = [1, 1, LeNet.INPUT_SIZE[-1], LeNet.INPUT_SIZE[-1]]
    config = get_basic_magnitude_sparsity_config(input_sample_size=input_sample_size)
    acc_aware_config = {
        "maximal_accuracy_degradation": max_accuracy_degradation,
        "initial_training_phase_epochs": initial_training_phase_epochs,
        "patience_epochs": patience_epochs,
    }
    config['compression']['accuracy_aware_training'] = acc_aware_config

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

    acc_aware_training_loop = PTAdaptiveCompressionTrainingLoop(config, compression_ctrl)
    model = acc_aware_training_loop.run(model,
                                        train_epoch_fn=train_fn,
                                        validate_fn=partial(validate_fn, train_loader=train_loader),
                                        configure_optimizers_fn=configure_optimizers_fn)
    assert compression_ctrl.compression_rate == pytest.approx(final_compression_rate, 1e-3)
    assert validate_fn(model, train_loader=train_loader) == pytest.approx(reference_final_metric, 1e-4)
