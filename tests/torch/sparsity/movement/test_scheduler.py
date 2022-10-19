from typing import List, Optional, Union
from unittest.mock import MagicMock

import nncf
import numpy as np
import pytest
from nncf.common.sparsity.schedulers import PolynomialThresholdScheduler
from pytest import approx


class SchedulerParams:
    def __init__(self, power: int = 3,
                 warmup_start_epoch: int = 1,
                 warmup_end_epoch: int = 3,
                 init_importance_threshold: float = -1.0,
                 final_importance_threshold: float = 0.0,
                 importance_regularization_factor: float = 0.1,
                 steps_per_epoch: Optional[int] = 4,
                 update_per_optimizer_step: bool = True):
        self.power = power
        self.warmup_start_epoch = warmup_start_epoch
        self.warmup_end_epoch = warmup_end_epoch
        self.init_importance_threshold = init_importance_threshold
        self.final_importance_threshold = final_importance_threshold
        self.importance_regularization_factor = importance_regularization_factor
        self.steps_per_epoch = steps_per_epoch
        self.update_per_optimizer_step = update_per_optimizer_step


@pytest.mark.parametrize('params,ref_threshold,ref_factor', [
    (SchedulerParams(2, 1, 3, -1, 0, 0.1, 4),  # normal use
     [-1., -1., -1., -1., -1., -0.7656, -0.5625, -0.3906, -0.2500, -0.1406, -0.0625, -0.0156, 0., 0., 0., 0., 0., 0., 0., 0.],
     [0., 0., 0., 0., 0., 0.0234, 0.0438, 0.0609, 0.0750, 0.0859, 0.0938, 0.0984, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
    (SchedulerParams(3, 0, 5, 1, 3, 1, 4, False),  # full range warmup, with update on epoch step
     [1., 1., 1., 1., 1.7718, 1.7718, 1.7718, 1.7718, 2.4508, 2.4508, 2.4508, 2.4508, 2.8178, 2.8178, 2.8178, 2.8178, 2.9688, 2.9688, 2.9688, 2.9688],
     [0., 0., 0., 0., 0.3859, 0.3859, 0.3859, 0.3859, 0.7254, 0.7254, 0.7254, 0.7254, 0.9089, 0.9089, 0.9089, 0.9089, 0.9844, 0.9844, 0.9844, 0.9844]),
    (SchedulerParams(4, 2, 8, 0, 5, 10, 3),  # warm up range overflow
     [0., 0., 0., 0., 0., 0., 0., 1.0219, 1.8785, 2.5887, 3.1702, 3.6396, 4.0123, 4.3027, 4.5237],
     [0., 0., 0., 0., 0., 0., 0., 2.0438, 3.7570, 5.1775, 6.3405, 7.2793, 8.0247, 8.6053, 9.0474]),
    (SchedulerParams(4, -1, 0, -1, 2, 5, 2),  # no warmup
     [2.] * 10,
     [5.] * 10),
])
def test_scheduler_can_produce_decayed_importance_threshold_and_regularization_factor(params: SchedulerParams, ref_threshold, ref_factor):
    scheduler = PolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
    threshold, factor = [], []
    for epoch in range(5):
        scheduler.epoch_step()
        for batch in range(params.steps_per_epoch):
            scheduler.step()
            threshold.append(scheduler.current_importance_threshold)
            factor.append(scheduler.current_importance_lambda)

    assert np.allclose(threshold, ref_threshold, atol=1e-4)
    assert np.allclose(factor, ref_factor, atol=1e-4)


def test_scheduler_get_state():
    params = SchedulerParams()
    scheduler = PolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
    assert scheduler.get_state() == {'current_epoch': -1,
                                     'current_step': -1,
                                     '_steps_per_epoch': params.steps_per_epoch}
    for epoch in range(4):
        scheduler.epoch_step()
        assert scheduler.get_state() == {'current_epoch': epoch,
                                         'current_step': epoch * params.steps_per_epoch - 1,
                                         '_steps_per_epoch': params.steps_per_epoch}
        for batch in range(params.steps_per_epoch):
            scheduler.step()
            assert scheduler.get_state() == {'current_epoch': epoch,
                                             'current_step': epoch * params.steps_per_epoch + batch,
                                             '_steps_per_epoch': params.steps_per_epoch}


@pytest.mark.parametrize('params', [
    SchedulerParams(steps_per_epoch=2),
    SchedulerParams(steps_per_epoch=4),
    SchedulerParams(steps_per_epoch=8),
    SchedulerParams(steps_per_epoch=None),
])
def test_scheduler_load_state(params):
    reload_step = 6
    steps_per_epoch = params.steps_per_epoch or 8  # check if we can resume 1st epoch even with `steps_per_epoch` not specified

    ref_scheduler = PolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
    ref_threshold, ref_factor = [], []
    for epoch in range(5):
        ref_scheduler.epoch_step()
        for batch in range(steps_per_epoch):
            ref_scheduler.step()
            if ref_scheduler.current_step > reload_step:
                ref_threshold.append(ref_scheduler.current_importance_threshold)
                ref_factor.append(ref_scheduler.current_importance_lambda)

    # check state dict is loaded
    scheduler = PolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
    ref_state = {'current_epoch': reload_step // steps_per_epoch,
                 'current_step': reload_step,
                 '_steps_per_epoch': params.steps_per_epoch}
    scheduler.load_state(ref_state)
    assert scheduler.current_epoch == ref_state['current_epoch']
    assert scheduler.current_step == ref_state['current_step']
    assert scheduler._steps_per_epoch == ref_state['_steps_per_epoch']

    # check can resume and continue
    # TODO: after resume, the threshold value before the very first `.step()` call is incorrect
    threshold, factor = [], []
    for rest_batch in range(steps_per_epoch - (reload_step + 1) % steps_per_epoch):
        scheduler.step()
        threshold.append(scheduler.current_importance_threshold)
        factor.append(scheduler.current_importance_lambda)
    for epoch in range(ref_state['current_epoch'] + 1, 5):
        scheduler.epoch_step()
        for batch in range(steps_per_epoch):
            scheduler.step()
            threshold.append(scheduler.current_importance_threshold)
            factor.append(scheduler.current_importance_lambda)
    assert np.allclose(threshold, ref_threshold)
    assert np.allclose(factor, ref_factor)


def test_scheduler_can_infer_steps_per_epoch():
    params = SchedulerParams(2, 1, 3, -1, 0, 0.1, steps_per_epoch=None)
    threshold_after_6_step_calls = approx(-0.7656, abs=1e-4)
    factor_after_6_step_calls = approx(0.0234, abs=1e-4)
    scheduler = PolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)

    scheduler.epoch_step()
    for _ in range(4):
        scheduler.step()
    assert scheduler._steps_per_epoch is None
    assert scheduler.current_importance_threshold == approx(params.init_importance_threshold)
    assert scheduler.current_importance_lambda == approx(0.)

    scheduler.epoch_step()
    assert scheduler._steps_per_epoch == 4
    scheduler.step()
    scheduler.step()
    assert scheduler.current_importance_threshold == threshold_after_6_step_calls
    assert scheduler.current_importance_lambda == factor_after_6_step_calls
