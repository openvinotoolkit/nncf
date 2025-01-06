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

import pytest

from nncf.common.sparsity.schedulers import MultiStepSparsityScheduler
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.sparsity.magnitude.test_helpers import get_basic_magnitude_sparsity_config
from tests.tensorflow.sparsity.magnitude.test_helpers import get_magnitude_test_model


def get_multistep_normed_abs_config():
    config = get_basic_magnitude_sparsity_config()
    compression_config = config["compression"]
    compression_config["params"] = {
        "schedule": "multistep",
        "weight_importance": "normed_abs",
        "multistep_steps": [1, 3],
        "multistep_sparsity_levels": [0.1, 0.5, 0.9],
    }
    return config


def test_magnitude_scheduler_can_do_epoch_step__with_norm():
    config = get_multistep_normed_abs_config()
    model = get_magnitude_test_model(config["input_info"]["sample_size"][1:])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, MultiStepSparsityScheduler)

    assert compression_ctrl.scheduler.current_sparsity_level == 0.1

    expected_levels = [0.1, 0.5, 0.5, 0.9]
    expected_thresholds = [0.219, 0.243, 0.243, 0.371]
    for expected_level, expected_threshold in zip(expected_levels, expected_thresholds):
        scheduler.epoch_step()
        assert compression_ctrl.scheduler.current_sparsity_level == expected_level
        nncf_stats = compression_ctrl.statistics()
        for layer_info in nncf_stats.magnitude_sparsity.thresholds:
            assert layer_info.threshold == pytest.approx(expected_threshold, 0.01)


def test_magnitude_scheduler_can_do_epoch_step__with_last():
    config = get_multistep_normed_abs_config()
    model = get_magnitude_test_model(config["input_info"]["sample_size"][1:])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler

    scheduler.epoch_step(3)
    assert compression_ctrl.scheduler.current_sparsity_level == 0.9
    nncf_stats = compression_ctrl.statistics()
    for layer_info in nncf_stats.magnitude_sparsity.thresholds:
        assert layer_info.threshold == pytest.approx(0.371, 0.01)

    scheduler.epoch_step()
    assert compression_ctrl.scheduler.current_sparsity_level == 0.9
    nncf_stats = compression_ctrl.statistics()
    for layer_info in nncf_stats.magnitude_sparsity.thresholds:
        assert layer_info.threshold == pytest.approx(0.371, 0.01)


def test_magnitude_scheduler_can_do_epoch_step__with_multistep():
    config = get_empty_config()
    config["compression"] = {
        "algorithm": "magnitude_sparsity",
        "params": {"schedule": "multistep", "multistep_steps": [1]},
    }
    model = get_magnitude_test_model(config["input_info"][0]["sample_size"][1:])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, MultiStepSparsityScheduler)

    assert compression_ctrl.scheduler.current_sparsity_level == 0.1
    assert scheduler.schedule.values == [0.1, 0.5]

    expected_levels = [0.1, 0.5, 0.5]
    for expected_level in expected_levels:
        scheduler.epoch_step()
        assert compression_ctrl.scheduler.current_sparsity_level == expected_level
