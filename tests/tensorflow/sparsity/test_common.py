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

from typing import List, Optional

import pytest

from nncf.common.sparsity.schedulers import AdaptiveSparsityScheduler
from nncf.common.sparsity.schedulers import ExponentialSparsityScheduler
from nncf.common.sparsity.schedulers import MultiStepSparsityScheduler
from nncf.common.sparsity.schedulers import PolynomialSparsityScheduler
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_conv_test_model
from tests.tensorflow.helpers import get_empty_config
from tests.tensorflow.helpers import get_mock_model


@pytest.mark.parametrize("algo", ("magnitude_sparsity", "rb_sparsity"))
@pytest.mark.parametrize(
    ("schedule_type", "scheduler_class"),
    (
        ("polynomial", PolynomialSparsityScheduler),
        ("exponential", ExponentialSparsityScheduler),
        ("multistep", MultiStepSparsityScheduler),
    ),
)
def test_can_choose_scheduler(algo, schedule_type, scheduler_class):
    config = get_empty_config()
    config["compression"] = {"algorithm": algo, "params": {"schedule": schedule_type}}
    _, compression_ctrl = create_compressed_model_and_algo_for_test(get_mock_model(), config)
    assert isinstance(compression_ctrl.scheduler, scheduler_class)


def test_can_not_create_rb_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config["compression"] = {"algorithm": "rb_sparsity", "params": {"schedule": "adaptive"}}
    with pytest.raises(NotImplementedError):
        _, _ = create_compressed_model_and_algo_for_test(get_mock_model(), config)


def test_can_not_create_magnitude_algo__with_adaptive_scheduler():
    config = get_empty_config()
    config["compression"] = {"algorithm": "magnitude_sparsity", "params": {"schedule": "adaptive"}}
    with pytest.raises(ValueError):
        _, _ = create_compressed_model_and_algo_for_test(get_mock_model(), config)


def get_poly_params():
    return {"power": 1, "sparsity_target_epoch": 2, "sparsity_target": 0.6, "sparsity_freeze_epoch": 4}


def get_multistep_params():
    return {"multistep_steps": [2, 3, 4], "multistep_sparsity_levels": [0.2, 0.4, 0.5, 0.6], "sparsity_freeze_epoch": 4}


@pytest.mark.parametrize("algo", ("magnitude_sparsity", "rb_sparsity"))
class TestSparseModules:
    def test_can_create_sparse_scheduler__with_defaults(self, algo):
        config = get_empty_config()

        config["compression"] = {"algorithm": algo, "params": {"schedule": "polynomial"}}
        _, compression_ctrl = create_compressed_model_and_algo_for_test(get_mock_model(), config)
        scheduler = compression_ctrl.scheduler
        assert scheduler.initial_level == 0
        assert scheduler.target_level == 0.5
        assert scheduler.target_epoch == 90
        assert scheduler.freeze_epoch == 100

    @pytest.mark.parametrize(
        ("schedule", "get_params", "ref_levels"),
        (
            ("polynomial", get_poly_params, [0.2, 0.2, 0.4, 0.6, 0.6, 0.6, 0.6]),
            ("exponential", get_poly_params, [0.2, 0.2, 0.4343145, 0.6, 0.6, 0.6, 0.6]),
            ("multistep", get_multistep_params, [0.2, 0.2, 0.2, 0.4, 0.5, 0.6, 0.6]),
        ),
    )
    def test_scheduler_can_do_epoch_step(self, algo, schedule, get_params, ref_levels):
        model = get_basic_conv_test_model()
        config = get_empty_config()
        config["compression"] = {
            "algorithm": algo,
            "sparsity_init": 0.2,
            "params": {**get_params(), "schedule": schedule},
        }

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


@pytest.fixture(name="magnitude_algo_mock")
def magnitude_algo_mock_(mocker):
    class MockSparsityAlgo:
        def __init__(self):
            self.set_sparsity_level = mocker.stub()
            self.freeze = mocker.stub()

        @property
        def current_sparsity_level(self) -> float:
            return 0.0

    return MockSparsityAlgo()


class TestPolynomialSparsityScheduler:
    @staticmethod
    def run_epoch(steps_per_epoch, scheduler, set_sparsity_mock, ref_vals, explicit, epoch):
        scheduler.epoch_step(epoch if explicit else None)
        if epoch < len(ref_vals) - 2:
            set_sparsity_mock.assert_called_once_with(ref_vals[epoch + 1])
        for i in range(steps_per_epoch):
            step = epoch * steps_per_epoch + i
            scheduler.step(step if explicit else None)
            set_sparsity_mock.reset_mock()

    @staticmethod
    def run_epoch_with_per_step_sparsity_check(
        steps_per_epoch: int,
        scheduler: PolynomialSparsityScheduler,
        set_sparsity_mock,
        ref_vals: List[Optional[float]],
        explicit,
        epoch,
    ):
        assert len(ref_vals) == steps_per_epoch
        scheduler.epoch_step(epoch if explicit else None)
        set_sparsity_mock.assert_not_called()
        for i in range(steps_per_epoch):
            step = epoch * steps_per_epoch + i
            scheduler.step(step if explicit else None)
            ref_sparsity_level = ref_vals[i]
            set_sparsity_mock.assert_called_once_with(pytest.approx(ref_sparsity_level))
            set_sparsity_mock.reset_mock()

    @pytest.mark.parametrize("explicit", [True, False], ids=["explicit_steps", "implicit_steps"])
    @pytest.mark.parametrize(
        "concavity_and_ref_sparsity_levels",
        [(False, [0.1, 0.1, 0.2, 0.5, 0.5]), (True, [pytest.approx(0.1), pytest.approx(0.1), 0.4, 0.5, 0.5])],
        ids=["convex", "concave"],
    )
    def test_polynomial_schedule_per_epoch_step(self, magnitude_algo_mock, concavity_and_ref_sparsity_levels, explicit):
        concave = concavity_and_ref_sparsity_levels[0]
        ref_sparsity_levels = concavity_and_ref_sparsity_levels[1]
        params = {
            "power": 2,
            "sparsity_init": 0.1,
            "sparsity_target": 0.5,
            "sparsity_target_epoch": 2,
            "sparsity_freeze_epoch": 3,
            "concave": concave,
        }

        scheduler = PolynomialSparsityScheduler(magnitude_algo_mock, params=params)
        mock = magnitude_algo_mock.set_sparsity_level

        steps_per_epoch = 3

        epoch = 0
        # After epoch 0
        self.run_epoch(steps_per_epoch, scheduler, mock, ref_sparsity_levels, explicit, epoch)
        epoch += 1

        # After epoch 1
        self.run_epoch(steps_per_epoch, scheduler, mock, ref_sparsity_levels, explicit, epoch)
        epoch += 1

        # After epoch 2
        self.run_epoch(steps_per_epoch, scheduler, mock, ref_sparsity_levels, explicit, epoch)
        assert scheduler.current_sparsity_level == ref_sparsity_levels[3]
        epoch += 1

        # After epoch 3 - sparsity freeze should occur
        self.run_epoch(steps_per_epoch, scheduler, mock, ref_sparsity_levels, explicit, epoch)
        magnitude_algo_mock.freeze.assert_called_once()
        assert scheduler.current_sparsity_level == ref_sparsity_levels[4]
        epoch += 1

        # After freezing
        for i in range(10):
            self.run_epoch(steps_per_epoch, scheduler, mock, ref_sparsity_levels, explicit, epoch + i)
            assert scheduler.current_sparsity_level == ref_sparsity_levels[4]

    @pytest.mark.parametrize("explicit", [True, False], ids=["explicit_steps", "implicit_steps"])
    def test_polynomial_schedule_per_optimizer_step(self, magnitude_algo_mock, explicit):
        steps_per_epoch = 3
        params = {
            "power": 2,
            "sparsity_init": 0.1,
            "sparsity_target": 0.5,
            "sparsity_target_epoch": 3,
            "sparsity_freeze_epoch": 4,
            "update_per_optimizer_step": True,
            "concave": False,
            "steps_per_epoch": steps_per_epoch,
        }

        scheduler = PolynomialSparsityScheduler(magnitude_algo_mock, params=params)
        mock = magnitude_algo_mock.set_sparsity_level

        per_epoch_ref_level_sequences = [
            [0.1, 85 / 810, 97 / 810],
            [117 / 810, 145 / 810, 181 / 810],
            [225 / 810, 277 / 810, 337 / 810],
            [0.5, 0.5, 0.5],
        ]

        for epoch, ref_level_sequence in enumerate(per_epoch_ref_level_sequences):
            self.run_epoch_with_per_step_sparsity_check(
                steps_per_epoch, scheduler, mock, ref_level_sequence, explicit, epoch
            )


@pytest.fixture(name="rb_algo_mock")
def rb_algo_mock_(mocker):
    class MockSparsityAlgo:
        def __init__(self):
            self.set_sparsity_level = mocker.stub()
            self.freeze = mocker.stub()
            from nncf.tensorflow.sparsity.rb.loss import SparseLoss

            self.loss = SparseLoss([])
            self.loss.current_sparsity = 0.3
            self.sparsity_init = 0

        def get_sparsity_init(self):
            return self.sparsity_init

        def set_sparsity_init(self, sparsity_init):
            self.sparsity_init = sparsity_init

        @property
        def current_sparsity_level(self) -> float:
            return self.loss.current_sparsity

    return MockSparsityAlgo()


class TestAdaptiveSparsityScheduler:
    @staticmethod
    def run_epoch(steps_per_epoch, scheduler, set_sparsity_mock):
        set_sparsity_mock.reset_mock()
        scheduler.epoch_step()
        for _ in range(steps_per_epoch):
            scheduler.step()

    @pytest.mark.parametrize(
        "ref_sparsity_levels", [([pytest.approx(x) for x in [0.25, 0.25, 0.3, 0.35, 0.4, 0.4, 0.4, 0.4, 0.4]])]
    )
    def test_adaptive_scheduler_per_epoch_step(self, rb_algo_mock, ref_sparsity_levels):
        params = {"sparsity_target": 0.4, "sparsity_target_epoch": 3, "sparsity_freeze_epoch": 7, "sparsity_init": 0.2}

        rb_algo_mock.set_sparsity_init(params["sparsity_init"])
        scheduler = AdaptiveSparsityScheduler(rb_algo_mock, params=params)
        mock = rb_algo_mock.set_sparsity_level

        steps_per_epoch = 3
        loss_current_sparsity = [0.3, 0.2, 0.22, 0.31, 0.34, 0.37, 0.48]

        for epoch_idx in range(7):
            rb_algo_mock.loss.current_sparsity = loss_current_sparsity[epoch_idx]
            self.run_epoch(steps_per_epoch, scheduler, mock)
            expected_level = ref_sparsity_levels[epoch_idx]
            mock.assert_called_once_with(expected_level)
