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

from nncf.common.sparsity.schedulers import PolynomialSparseScheduler, ExponentialSparsityScheduler, \
    MultiStepSparsityScheduler
from beta.tests.tensorflow.helpers import get_basic_conv_test_model, get_empty_config, \
    create_compressed_model_and_algo_for_test, get_mock_model


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity',))
@pytest.mark.parametrize(('schedule_type', 'scheduler_class'),
                         (
                             ('polynomial', PolynomialSparseScheduler),
                             ('exponential', ExponentialSparsityScheduler),
                             ('multistep', MultiStepSparsityScheduler)
                         ))


def test_can_choose_scheduler(algo, schedule_type, scheduler_class):
    config = get_empty_config()
    config['compression'] = Dict({'algorithm': algo, "params": {"schedule": schedule_type}})
    _, compression_ctrl = create_compressed_model_and_algo_for_test(get_mock_model(), config)
    assert isinstance(compression_ctrl.scheduler, scheduler_class)


def get_poly_params():
    return {
        'power': 1, 'sparsity_target_epoch': 2, 'sparsity_target': 0.6,
        'sparsity_freeze_epoch': 4
    }


def get_multistep_params():
    return {
        'multistep_steps': [2, 3, 4],
        'multistep_sparsity_levels': [0.2, 0.4, 0.5, 0.6],
        'sparsity_freeze_epoch': 4
    }


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity',))
class TestSparseModules:
    def test_can_create_sparse_scheduler__with_defaults(self, algo):
        config = get_empty_config()

        config['compression'] = Dict({'algorithm': algo, "params": {"schedule": 'polynomial'}})
        _, compression_ctrl = create_compressed_model_and_algo_for_test(get_mock_model(), config)
        scheduler = compression_ctrl.scheduler
        assert scheduler.initial_sparsity == 0
        assert scheduler.target_sparsity == 0.5
        assert scheduler.target_epoch == 90
        assert scheduler.freeze_epoch == 100

    @pytest.mark.parametrize(('schedule', 'get_params', 'ref_levels'),
                             (('polynomial', get_poly_params, [0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6]),
                              ('exponential', get_poly_params, [0.2, 0.2, 0.4343145, 0.6, 0.6, 0.6, 0.6]),
                              ('multistep', get_multistep_params, [0.2, 0.2, 0.2, 0.4, 0.5, 0.6, 0.6])))
    def test_scheduler_can_do_epoch_step(self, algo, schedule, get_params, ref_levels):
        model = get_basic_conv_test_model()
        config = get_empty_config()
        config['compression'] = Dict(
            {'algorithm': algo, 'sparsity_init': 0.2, 'params': {**get_params(), 'schedule': schedule}})

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        scheduler = compression_ctrl.scheduler

        assert pytest.approx(scheduler.current_sparsity_level) == ref_levels[0]
        for ref_level in ref_levels[1:]:
            scheduler.epoch_step()
            assert pytest.approx(scheduler.current_sparsity_level) == ref_level

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
        scheduler = compression_ctrl.scheduler

        assert pytest.approx(scheduler.current_sparsity_level) == ref_levels[0]
        for i, ref_level in enumerate(ref_levels[1:]):
            scheduler.epoch_step(i)
            assert pytest.approx(scheduler.current_sparsity_level) == ref_level
