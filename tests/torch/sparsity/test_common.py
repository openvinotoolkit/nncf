"""
 Copyright (c) 2019-2022 Intel Corporation
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
from typing import List
from typing import Optional

import pytest
import torch

from nncf import NNCFConfig
from nncf.api.compression import CompressionStage
from nncf.common.compression import BaseCompressionAlgorithmController as BaseController
from nncf.common.compression import BaseControllerStateNames
from nncf.common.sparsity.schedulers import AdaptiveSparsityScheduler
from nncf.common.sparsity.schedulers import ExponentialSparsityScheduler
from nncf.common.sparsity.schedulers import MultiStepSparsityScheduler
from nncf.common.sparsity.schedulers import PolynomialSparsityScheduler
from tests.torch.helpers import BasicConvTestModel
from tests.torch.helpers import MockModel
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test
from tests.torch.helpers import fill_params_of_model_by_normal
from tests.torch.helpers import get_empty_config


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
@pytest.mark.parametrize(('schedule_type', 'scheduler_class'),
                         (
                             ('polynomial', PolynomialSparsityScheduler),
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


@pytest.mark.skip()
def test_can_not_create_magnitude_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config['compression'] = {'algorithm': 'magnitude_sparsity', "params": {"schedule": 'adaptive'}}
    with pytest.raises(TypeError):
        _, _ = create_compressed_model_and_algo_for_test(MockModel(), config)


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
                         ('magnitude_sparsity', 'rb_sparsity'))
class TestSparseModules:
    def test_can_create_sparse_scheduler__with_defaults(self, algo):
        config = get_empty_config()
        config['compression'] = {'algorithm': algo, "params": {"schedule": 'polynomial'}}
        _, compression_ctrl = create_compressed_model_and_algo_for_test(MockModel(), config)
        scheduler = compression_ctrl.scheduler
        assert scheduler.initial_level == 0
        assert scheduler.target_level == 0.5
        assert scheduler.target_epoch == 90
        assert scheduler.freeze_epoch == 100

    def test_compression_ctrl_state(self, algo):
        config = get_empty_config()
        config['compression'] = {'algorithm': algo, "params": {"schedule": 'polynomial'}}
        _, ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config)

        assert ctrl.scheduler.current_step == -1
        assert ctrl.scheduler.current_epoch == -1

        # Test get state
        ctrl.scheduler.load_state({'current_step': 100, 'current_epoch': 5})
        compression_state = ctrl.get_compression_state()
        saved_ctrl_state = compression_state[BaseController.CONTROLLER_STATE]
        assert saved_ctrl_state == ctrl.get_state()
        algo_state = next(iter(saved_ctrl_state.values()))
        assert algo_state == {
            BaseControllerStateNames.COMPRESSION_STAGE: CompressionStage.PARTIALLY_COMPRESSED,
            BaseControllerStateNames.SCHEDULER: {'current_step': 100, 'current_epoch': 5},
            BaseControllerStateNames.LOSS: None
        }

        # Test load state
        _, ctrl = create_compressed_model_and_algo_for_test(BasicConvTestModel(), config,
                                                            compression_state=compression_state)
        assert ctrl.scheduler.current_step == 100
        assert ctrl.scheduler.current_epoch == 5
        compression_state = ctrl.get_compression_state()
        loaded_ctrl_state = compression_state[BaseController.CONTROLLER_STATE]
        assert loaded_ctrl_state == ctrl.get_state()
        assert loaded_ctrl_state == saved_ctrl_state

    @pytest.mark.parametrize(('schedule', 'get_params', 'ref_levels'),
                             (('polynomial', get_poly_params, [0.2, 0.4, 0.6, 0.6, 0.6, 0.6]),
                              ('exponential', get_poly_params, [0.2, 0.4343145, 0.6, 0.6, 0.6, 0.6]),
                              ('multistep', get_multistep_params, [0.2, 0.2, 0.4, 0.5, 0.6, 0.6])))
    def test_scheduler_can_do_epoch_step(self, algo, schedule, get_params, ref_levels):
        model = BasicConvTestModel()
        config = get_empty_config()
        config['compression'] = {'algorithm': algo,
                                 "sparsity_init": 0.2, "params": {**get_params(), "schedule": schedule}}

        _, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)

        scheduler = compression_ctrl.scheduler
        scheduler.epoch_step()
        assert pytest.approx(scheduler.current_sparsity_level) == ref_levels[0]
        for ref_level in ref_levels[1:]:
            scheduler.epoch_step()
            assert pytest.approx(scheduler.current_sparsity_level) == ref_level

        for m in compression_ctrl.sparsified_module_info:
            if hasattr(m.operand, "frozen"):
                assert m.operand.frozen


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
        set_sparsity_mock.reset_mock()
        scheduler.epoch_step()
        for _ in range(steps_per_epoch):
            scheduler.step()

    @staticmethod
    def run_epoch_with_per_step_sparsity_check(steps_per_epoch: int, scheduler: PolynomialSparsityScheduler,
                                               set_sparsity_mock,
                                               ref_vals: List[Optional[float]]):
        assert len(ref_vals) == steps_per_epoch + 1
        scheduler.epoch_step()
        set_sparsity_mock.assert_not_called()
        for i in range(steps_per_epoch):
            scheduler.step()
            ref_sparsity_level = ref_vals[i + 1]
            if ref_sparsity_level is not None:
                set_sparsity_mock.assert_called_once_with(pytest.approx(ref_sparsity_level))
            else:
                set_sparsity_mock.assert_not_called()
            set_sparsity_mock.reset_mock()

    @pytest.mark.parametrize("concavity_and_ref_sparsity_levels", [
        (False, [0.1, pytest.approx(13 / 90), pytest.approx(25 / 90), 0.5, 0.5]),
        (True, [pytest.approx(0.1), pytest.approx(29 / 90), pytest.approx(41 / 90), 0.5, 0.5])],
                             ids=["concave", "convex"])
    def test_polynomial_schedule_per_epoch_step(self, magnitude_algo_mock, concavity_and_ref_sparsity_levels):
        concave = concavity_and_ref_sparsity_levels[0]
        ref_sparsity_levels = concavity_and_ref_sparsity_levels[1]

        params = {
            "power": 2,
            'sparsity_init': 0.1,
            'sparsity_target': 0.5,
            "sparsity_target_epoch": 3,
            "sparsity_freeze_epoch": 4,
            "concave": concave,
        }

        scheduler = PolynomialSparsityScheduler(magnitude_algo_mock, params=params)
        mock = magnitude_algo_mock.set_sparsity_level

        steps_per_epoch = 3

        # epoch 0
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[0])

        # epoch 1
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[1])

        # epoch 2
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[2])
        assert scheduler.current_sparsity_level == ref_sparsity_levels[2]

        # epoch 3
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[3])
        assert scheduler.current_sparsity_level == ref_sparsity_levels[3]

        # epoch 4 - sparsity freeze should occur
        self.run_epoch(steps_per_epoch, scheduler, mock)
        mock.assert_called_once_with(ref_sparsity_levels[4])
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
            'sparsity_init': 0.1,
            'sparsity_target': 0.5,
            "sparsity_target_epoch": 3,
            "sparsity_freeze_epoch": 4,
            "update_per_optimizer_step": True,
            "concave": False
        }

        if specify_steps_per_epoch_in_config:
            params["steps_per_epoch"] = steps_per_epoch

        scheduler = PolynomialSparsityScheduler(magnitude_algo_mock, params=params)
        mock = magnitude_algo_mock.set_sparsity_level
        mock.reset_mock()
        no_update_ref_sparsity_level_sequence = [0.1, None, None, None]
        first_sparsity_level_sequence = [0.1, 0.1, 85 / 810, 97 / 810]
        second_sparsity_level_sequence = [13 / 90, 13 / 90, 145 / 810, 181 / 810]

        if specify_steps_per_epoch_in_config:
            per_epoch_ref_level_sequences = [first_sparsity_level_sequence, second_sparsity_level_sequence]
        else:
            per_epoch_ref_level_sequences = [no_update_ref_sparsity_level_sequence, first_sparsity_level_sequence]

        for ref_level_sequence in per_epoch_ref_level_sequences:
            self.run_epoch_with_per_step_sparsity_check(steps_per_epoch,
                                                        scheduler, mock,
                                                        ref_level_sequence)


@pytest.fixture(name="rb_algo_mock")
def rb_algo_mock_(mocker):
    class MockSparsityAlgo:
        def __init__(self):
            self.set_sparsity_level = mocker.stub()
            self.freeze = mocker.stub()
            from nncf.torch.sparsity.rb.loss import SparseLoss
            self.loss = SparseLoss()
            self.loss.current_sparsity = 0.3

    return MockSparsityAlgo()


class TestAdaptiveSparsityScheduler:
    @staticmethod
    def run_epoch(steps_per_epoch, scheduler, set_sparsity_mock):
        set_sparsity_mock.reset_mock()
        scheduler.epoch_step()
        for _ in range(steps_per_epoch):
            scheduler.step()

    @pytest.mark.parametrize("ref_sparsity_levels", [([0.25, 0.25, 0.3, 0.35, 0.4, 0.4, 0.4])])
    def test_adaptive_scheduler_per_epoch_step(self, rb_algo_mock, ref_sparsity_levels):
        params = {
            'sparsity_init': 0.2,
            'sparsity_target': 0.4,
            "sparsity_target_epoch": 3,
            "sparsity_freeze_epoch": 7
        }

        scheduler = AdaptiveSparsityScheduler(rb_algo_mock, params=params)
        mock = rb_algo_mock.set_sparsity_level

        steps_per_epoch = 3
        loss_current_sparsity = [0.3, 0.2, 0.22, 0.31, 0.34, 0.37, 0.48]

        for epoch_idx in range(7):
            rb_algo_mock.loss.current_sparsity = loss_current_sparsity[epoch_idx]
            self.run_epoch(steps_per_epoch, scheduler, mock)
            expected_level = ref_sparsity_levels[epoch_idx]
            mock.assert_called_once_with(pytest.approx(expected_level))


REF_DEFAULT_STATE = {
    PolynomialSparsityScheduler: {'current_step': -1, 'current_epoch': -1, '_steps_per_epoch': None},
    ExponentialSparsityScheduler: {'current_step': -1, 'current_epoch': -1},
    MultiStepSparsityScheduler: {'current_step': -1, 'current_epoch': -1},
    AdaptiveSparsityScheduler: {'current_step': -1, 'current_epoch': -1,
                                'num_bad_epochs': 0, 'current_sparsity_level': 0.3}
}


class MockLoss:
    def __init__(self):
        self.current_sparsity = 0.5


class MockCompressionController:
    def __init__(self):
        self.loss = MockLoss()

    def set_sparsity_level(self, level):
        pass


@pytest.mark.parametrize('scheduler_cls', [PolynomialSparsityScheduler, ExponentialSparsityScheduler,
                                           MultiStepSparsityScheduler, AdaptiveSparsityScheduler],
                         ids=['Polynomial', 'Exponential', 'Multistep', 'Adaptive'])
def test_scheduler_get_state(scheduler_cls):
    args = (MockCompressionController(), {'sparsity_init': 0.3, 'update_per_optimizer_step': True, 'patience': 2})
    scheduler = scheduler_cls(*args)

    # Test init state
    assert scheduler.get_state() == REF_DEFAULT_STATE[scheduler_cls]

    for _ in range(5):
        scheduler.step()
    scheduler.epoch_step()

    # Test get state
    state = scheduler.get_state()
    assert state['current_step'] == 4
    assert state['current_epoch'] == 0
    if scheduler_cls == PolynomialSparsityScheduler:
        assert state['_steps_per_epoch'] == 5
    if scheduler_cls == AdaptiveSparsityScheduler:
        assert state['num_bad_epochs'] == 1

    # Test load state
    new_scheduler = scheduler_cls(*args)
    new_scheduler.load_state(state)
    assert new_scheduler.get_state() == scheduler.get_state()
    assert new_scheduler.current_step == 4
    assert new_scheduler.current_epoch == 0
    if scheduler_cls == PolynomialSparsityScheduler:
        # pylint: disable=protected-access
        assert new_scheduler._steps_per_epoch == 5
    if scheduler_cls == AdaptiveSparsityScheduler:
        assert new_scheduler.num_bad_epochs == 1
        assert new_scheduler.current_sparsity_level == pytest.approx(0.3)


@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
def test_sparsity_statistics_add_module(algo):
    config = get_empty_config()
    sparsity_init = 0.5
    config['compression'] = {'algorithm': algo, 'sparsity_init': sparsity_init}
    model = TwoConvTestModel()
    fill_params_of_model_by_normal(model)
    submodule = TwoConvTestModel()
    fill_params_of_model_by_normal(submodule)
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, config)
    statistics_before = compression_ctrl.statistics()
    model.add_module('submodule', submodule)
    statistics_after = compression_ctrl.statistics()
    assert getattr(statistics_before, algo).model_statistics.sparsity_level == \
           getattr(statistics_after, algo).model_statistics.sparsity_level
    assert getattr(statistics_before, algo).model_statistics.sparsity_level_for_layers == \
           getattr(statistics_after, algo).model_statistics.sparsity_level_for_layers


class EmbeddingOnlyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)

    def forward(self, x):
        return self.embedding(x)

@pytest.mark.parametrize('algo',
                         ('magnitude_sparsity', 'rb_sparsity'))
def test_can_sparsify_embedding(algo):
    config = {
        "input_info": {
            "sample_size": [1, 10],
            "type": "long"
        }
    }
    sparsity_init = 0.5
    config['compression'] = {'algorithm': algo, 'sparsity_init': sparsity_init}
    nncf_config = NNCFConfig.from_dict(config)
    model = EmbeddingOnlyModel()
    model, compression_ctrl = create_compressed_model_and_algo_for_test(model, nncf_config)

    # Should pass
    _ = compression_ctrl.statistics()
