"""
 Copyright (c) 2020 Intel Corporation
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
from addict import Dict

from nncf.common.sparsity.schedulers import MultiStepSparsityScheduler
from beta.tests.tensorflow.helpers import get_empty_config, create_compressed_model_and_algo_for_test
from beta.tests.tensorflow.sparsity.magnitude.test_helpers import get_magnitude_test_model, \
    get_basic_magnitude_sparsity_config


def get_multistep_normed_abs_config():
    config = get_basic_magnitude_sparsity_config()
    compression_config = config['compression']
    compression_config['params'] = {
        'schedule': 'multistep',
        'weight_importance': 'normed_abs',
        'multistep_steps': [1, 3],
        'multistep_sparsity_levels': [0.1, 0.5, 0.9]
    }
    return config


def test_magnitude_scheduler_can_do_epoch_step__with_norm():
    config = get_multistep_normed_abs_config()
    model = get_magnitude_test_model(config['input_info']['sample_size'][1:])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, MultiStepSparsityScheduler)

    scheduler.epoch_step()
    assert pytest.approx(compression_ctrl.sparsity_level) == 0.1
    assert compression_ctrl.statistics()['sparsity_threshold'] == pytest.approx(0.219, 0.01)

    scheduler.epoch_step()
    assert compression_ctrl.sparsity_level == 0.5
    assert compression_ctrl.statistics()['sparsity_threshold'] == pytest.approx(0.243, 0.01)

    scheduler.epoch_step()
    assert compression_ctrl.sparsity_level == 0.5
    assert compression_ctrl.statistics()['sparsity_threshold'] == pytest.approx(0.243, 0.01)

    scheduler.epoch_step()
    assert compression_ctrl.sparsity_level == 0.9
    assert compression_ctrl.statistics()['sparsity_threshold'] == pytest.approx(0.371, 0.01)


def test_magnitude_scheduler_can_do_epoch_step__with_last():
    config = get_multistep_normed_abs_config()
    model = get_magnitude_test_model(config['input_info']['sample_size'][1:])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler

    scheduler.epoch_step(3)
    assert compression_ctrl.sparsity_level == 0.9
    assert compression_ctrl.statistics()['sparsity_threshold'] == pytest.approx(0.371, 0.01)

    scheduler.epoch_step()
    assert compression_ctrl.sparsity_level == 0.9
    assert compression_ctrl.statistics()['sparsity_threshold'] == pytest.approx(0.371, 0.01)


def test_magnitude_scheduler_can_do_epoch_step__with_multistep():
    config = get_empty_config()
    config["compression"] = Dict({"algorithm": "magnitude_sparsity",
                                  "params": {"schedule": "multistep", 'multistep_steps': [1]}})
    model = get_magnitude_test_model(config['input_info'][0]['sample_size'][1:])
    _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    scheduler = compression_ctrl.scheduler
    assert isinstance(scheduler, MultiStepSparsityScheduler)
    assert scheduler.sparsity_levels == [0.1, 0.5]
    scheduler.epoch_step()
    assert pytest.approx(compression_ctrl.sparsity_level) == 0.1

    scheduler.epoch_step()
    assert compression_ctrl.sparsity_level == 0.5

    scheduler.epoch_step()
    assert compression_ctrl.sparsity_level == 0.5
