"""
 Copyright (c) 2019-2020 Intel Corporation
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
from typing import List, Optional

import pytest

from nncf.sparsity.schedulers import PolynomialSparseScheduler, ExponentialSparsityScheduler, \
    AdaptiveSparsityScheduler, MultiStepSparsityScheduler
from tests.helpers import BasicConvTestModel, get_empty_config, create_compressed_model_and_algo_for_test, \
    MockModel


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
@pytest.mark.parametrize(('schedule_type', 'scheduler_class'),
                         (
                             ('polynomial', PolynomialSparseScheduler),
                             ('exponential', ExponentialSparsityScheduler),
                             ('multistep', MultiStepSparsityScheduler)
                         ))


def test_can_choose_scheduler(algo, schedule_type, scheduler_class):
    config = get_empty_config()
    config['compression'] = {'algorithm': algo, "params": {"schedule": schedule_type}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert isinstance(compression_ctrl.scheduler, scheduler_class)


def test_can_create_rb_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config['compression'] = {'algorithm': 'rb_sparsity', "params": {"schedule": 'adaptive'}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
    assert isinstance(compression_ctrl.scheduler, AdaptiveSparsityScheduler)


def test_can_not_create_magnitude_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config['compression'] = {'algorithm': 'magnitude_sparsity', "params": {"schedule": 'adaptive'}}
    with pytest.raises(TypeError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


def get_poly_params():
    return {
        'power': 1, 'sparsity_target_epoch': 2, 'sparsity_init': 0.2, 'sparsity_target': 0.6,
        'sparsity_freeze_epoch': 4
    }


def get_multistep_params():
    return {
        'multistep_steps': [2, 3, 4],
        'multistep_sparsity_levels': [0.2, 0.4, 0.5, 0.6],
        'sparsity_freeze_epoch': 4
    }


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
class TestSparseModules:
    def test_can_create_sparse_scheduler__with_defaults(self, algo):
        config = get_empty_config()

        config['compression'] = {'algorithm': algo, "params": {"schedule": 'polynomial'}}
        _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
        scheduler = compression_ctrl.scheduler
        assert scheduler.initial_sparsity == 0
        assert scheduler.sparsity_target == 0.5
        assert scheduler.sparsity_target_epoch == 90
        assert scheduler.sparsity_freeze_epoch == 100

    @pytest.mark.parametrize(('schedule', 'get_params', 'ref_levels'),
                             (('polynomial', get_poly_params, [0.2, 0.4, 0.6, 0.6, 0.6, 0.6]),
                              ('exponential', get_poly_params, [0.2, 0.4343145, 0.6, 0.6, 0.6, 0.6]),
                              ('multistep', get_multistep_params, [0.2, 0.2, 0.4, 0.5, 0.6, 0.6])))
    def test_scheduler_can_do_epoch_step(self, algo, schedule, get_params, ref_levels):
        model = BasicConvTestModel()
        config = get_empty_config()
        config['compression'] = {'algorithm': algo, "params": {**get_params(), "schedule": schedule}}

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        scheduler = compression_ctrl.scheduler

        assert pytest.approx(scheduler.current_sparsity_level) == ref_levels[0]
        for ref_level in ref_levels[1:]:
            scheduler.epoch_step()
            assert pytest.approx(scheduler.current_sparsity_level) == ref_level

        for m in compression_ctrl.sparsified_module_info:
            if hasattr(m.operand, "sparsify"):
                assert not m.operand.sparsify


@pytest.fixture(name="magnitude_algo_mock")
def magnitude_algo_mock_(mocker):
    class MockSparsityAlgo:
        def __init__(self):
            self.set_sparsity_level = mocker.stub()
            self.freeze = mocker.stub()
    return MockSparsityAlgo()


class TestPolynomialSparsityScheduler:
    @staticmethod
    def run_epoch(steps_per_epoch, scheduler, set_sparsity_mock):
        for _ in range(steps_per_epoch):
            scheduler.step()
            set_sparsity_mock.reset_mock()
        scheduler.epoch_step()

    @staticmethod
    def run_epoch_with_per_step_sparsity_check(steps_per_epoch: int, scheduler: PolynomialSparseScheduler,
                                               set_sparsity_mock,
                                               ref_vals: List[Optional[float]]):
        assert len(ref_vals) == steps_per_epoch
        for i in range(steps_per_epoch):
            scheduler.step()
            ref_sparsity_level = ref_vals[i]
            if ref_sparsity_level is not None:
                set_sparsity_mock.assert_called_once_with(pytest.approx(ref_sparsity_level))
            else:
                set_sparsity_mock.assert_not_called()
            set_sparsity_mock.reset_mock()
        scheduler.epoch_step()

    REF_SPARSITY_LEVELS_PER_CONCAVE_MODE = [
        (True, [0.1, pytest.approx(13 / 90), pytest.approx(25 / 90), 0.5, 0.5]),
        (False, [0.1, pytest.approx(29 / 90), pytest.approx(41 / 90), 0.5, 0.5])
    ]

    @pytest.mark.parametrize("concavity_and_ref_sparsity_levels", [
        (True, [0.1, pytest.approx(13 / 90), pytest.approx(25 / 90), 0.5, 0.5]),
        (False, [pytest.approx(0.1), pytest.approx(29 / 90), pytest.approx(41 / 90), 0.5, 0.5])],
                             ids=["concave", "convex"])
    def test_polynomial_schedule_per_epoch_step(self, magnitude_algo_mock, concavity_and_ref_sparsity_levels):
        concave = concavity_and_ref_sparsity_levels[0]
        ref_sparsity_levels = concavity_and_ref_sparsity_levels[1]

        params = {
            "power": 2,
            "sparsity_init": 0.1,
            'sparsity_target': 0.5,
            "sparsity_target_epoch": 3,
            "sparsity_freeze_epoch": 4,
            "concave": concave,
        }
        scheduler = PolynomialSparseScheduler(magnitude_algo_mock, params=params)
        mock = magnitude_algo_mock.set_sparsity_level
        mock.assert_called_once_with(ref_sparsity_levels[0])
        mock.reset_mock()

        steps_per_epoch = 3


        # After epoch 0
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[1])

        # After epoch 1
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[2])

        # After epoch 2
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[3])
        assert scheduler.current_sparsity_level == ref_sparsity_levels[3]

        # After epoch 3 - sparsity freeze should occur
        self.run_epoch(steps_per_epoch, scheduler, mock)
        magnitude_algo_mock.freeze.assert_called_once()
        assert scheduler.current_sparsity_level == ref_sparsity_levels[4]

        # After freezing
        for _ in range(10):
            self.run_epoch(steps_per_epoch, scheduler, mock)
            assert scheduler.current_sparsity_level == ref_sparsity_levels[4]

    @pytest.mark.parametrize("specify_steps_per_epoch_in_config", [True, False])
    def test_polynomial_schedule_per_optimizer_step(self, magnitude_algo_mock, specify_steps_per_epoch_in_config):
        steps_per_epoch = 3
        params = {
            "power": 2,
            "sparsity_init": 0.1,
            'sparsity_target': 0.5,
            "sparsity_target_epoch": 3,
            "sparsity_freeze_epoch": 4,
            "update_per_optimizer_step": True,
            "concave": True
        }

        if specify_steps_per_epoch_in_config:
            params["steps_per_epoch"] = steps_per_epoch

        scheduler = PolynomialSparseScheduler(magnitude_algo_mock, params=params)
        mock = magnitude_algo_mock.set_sparsity_level
        mock.assert_called_once_with(pytest.approx(0.1))
        mock.reset_mock()

        no_update_ref_sparsity_level_sequence = [None, None, None]
        first_sparsity_level_sequence = [85 / 810, 97 / 810, 13 / 90]
        second_sparsity_level_sequence = [145 / 810, 181 / 810, 25 / 90]

        if specify_steps_per_epoch_in_config:
            per_epoch_ref_level_sequences = [first_sparsity_level_sequence, second_sparsity_level_sequence]
        else:
            per_epoch_ref_level_sequences = [no_update_ref_sparsity_level_sequence, first_sparsity_level_sequence]

        for ref_level_sequence in per_epoch_ref_level_sequences:
            self.run_epoch_with_per_step_sparsity_check(steps_per_epoch,
                                                        scheduler, mock,
                                                        ref_level_sequence)
            last_ref_level = ref_level_sequence[-1]
            if last_ref_level is not None:
                mock.assert_called_once_with(pytest.approx(last_ref_level))
            mock.reset_mock()
