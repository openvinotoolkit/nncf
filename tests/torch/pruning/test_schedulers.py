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
import numpy as np
import pytest

from nncf.common.pruning.schedulers import BaselinePruningScheduler
from nncf.common.pruning.schedulers import ExponentialWithBiasPruningScheduler
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.pruning.helpers import PruningTestModel
from tests.torch.pruning.helpers import get_pruning_baseline_config
from tests.torch.pruning.helpers import get_pruning_exponential_config


def test_baseline_scheduler():
    """
    Test baseline scheduler parameters and changes of params during epochs.
    """
    config = get_pruning_baseline_config()
    config["compression"]["algorithm"] = "filter_pruning"
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler

    # Check default params
    assert isinstance(scheduler, BaselinePruningScheduler)
    assert pytest.approx(scheduler.target_level) == 0.5
    assert pytest.approx(scheduler.initial_level) == 0.0
    assert scheduler.num_warmup_epochs == 1

    # Check pruning params before epoch 0
    scheduler.epoch_step()
    assert pytest.approx(scheduler.current_pruning_level) == 0.0
    assert pytest.approx(compression_ctrl.pruning_level) == 0.0
    assert scheduler.current_epoch == 0
    assert compression_ctrl.frozen is False

    # Check pruning params after epoch 0
    scheduler.epoch_step()
    assert pytest.approx(scheduler.current_pruning_level) == 0.5
    assert pytest.approx(compression_ctrl.pruning_level) == 0.5
    assert scheduler.current_epoch == 1
    assert compression_ctrl.frozen is True


def test_exponential_scheduler():
    """
    Test exponential with bias scheduler parameters and changes of params during epochs.
    """
    config = get_pruning_exponential_config()
    config["compression"]["algorithm"] = "filter_pruning"
    model = PruningTestModel()
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler

    # Check default params
    assert isinstance(scheduler, ExponentialWithBiasPruningScheduler)
    assert pytest.approx(scheduler.target_level) == 0.5
    assert pytest.approx(scheduler.initial_level) == 0.0
    assert scheduler.num_warmup_epochs == 1
    assert scheduler.num_pruning_epochs == 20
    assert pytest.approx(scheduler.a, abs=1e-4) == -0.5
    assert pytest.approx(scheduler.b, abs=1e-4) == 0.5
    assert pytest.approx(scheduler.k, abs=1e-4) == 0.5836

    # Check pruning params before epoch 0
    scheduler.epoch_step()
    assert pytest.approx(scheduler.current_pruning_level) == 0.0
    assert pytest.approx(compression_ctrl.pruning_level) == 0.0
    assert compression_ctrl.frozen is False
    assert scheduler.current_epoch == 0

    # Check pruning params on epoch 1 - 19
    for i in range(20):
        # Check pruning params on epoch 1
        scheduler.epoch_step()
        pruning_level = (
            scheduler.a * np.exp(-scheduler.k * (scheduler.current_epoch - scheduler.num_warmup_epochs)) + scheduler.b
        )
        assert pytest.approx(scheduler.current_pruning_level) == pruning_level
        assert pytest.approx(compression_ctrl.pruning_level) == pruning_level
        assert compression_ctrl.frozen is False
        assert scheduler.current_epoch == i + 1

    # Check pruning params after epoch 20
    scheduler.epoch_step()
    assert pytest.approx(scheduler.current_pruning_level, abs=1e-4) == 0.5
    assert pytest.approx(compression_ctrl.pruning_level, abs=1e-4) == 0.5
    assert compression_ctrl.frozen is True
    assert scheduler.current_epoch == 21


@pytest.fixture(name="pruning_controller_mock")
def pruning_controller_mock_(mocker):
    class MockPruningController:
        def __init__(self):
            self.pruning_init = 0
            self.prune_flops = False
            self.set_pruning_level = mocker.stub()
            self.freeze = mocker.stub()
            self.step = mocker.stub()

        def set_pruning_init(self, pruning_init):
            self.pruning_init = pruning_init

    return MockPruningController()


def test_exponential_with_bias(pruning_controller_mock):
    pruning_init = 0.1
    scheduler_params = {"pruning_target": 0.7, "num_init_steps": 3, "pruning_steps": 5}
    expected_levels = [0, 0, 0, 0.1, 0.6489741, 0.6956869, 0.6996617, 0.7, 0.7, 0.7]
    freeze_epoch = scheduler_params["num_init_steps"] + scheduler_params["pruning_steps"]

    pruning_controller_mock.set_pruning_init(pruning_init)
    scheduler = ExponentialWithBiasPruningScheduler(pruning_controller_mock, scheduler_params)

    num_epochs = 10
    steps_per_epoch = 3

    assert len(expected_levels) == num_epochs
    for epoch_idx in range(num_epochs):
        scheduler.epoch_step()
        expected_level = pytest.approx(expected_levels[epoch_idx])
        assert scheduler.current_pruning_level == expected_level
        for _ in range(steps_per_epoch):
            scheduler.step()

        pruning_controller_mock.set_pruning_level.assert_called_once_with(expected_level)
        pruning_controller_mock.set_pruning_level.reset_mock()

        if epoch_idx < freeze_epoch:
            pruning_controller_mock.freeze.assert_not_called()
        else:
            pruning_controller_mock.freeze.assert_called_once()
            pruning_controller_mock.freeze.reset_mock()
