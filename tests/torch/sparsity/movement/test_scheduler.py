from typing import List, Optional, Union
from unittest.mock import MagicMock, Mock
from collections import defaultdict

import nncf
import numpy as np
import pytest
from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from tests.torch.sparsity.movement.helpers import SchedulerParams
from pytest import approx


desc_test_decayed_importance_threshold_and_regularization_factor = {
    "normal_warmup_range": dict(
        params=SchedulerParams(power=2,
                               warmup_start_epoch=1, warmup_end_epoch=3,
                               init_importance_threshold=-1, final_importance_threshold=0,
                               importance_regularization_factor=0.1, steps_per_epoch=4,
                               enable_structured_masking=False),
        ref_threshold=[-1., -1., -1., -1., -1., -0.7656, -0.5625, -0.3906, -0.2500, -0.1406,
                       -0.0625, -0.0156, 0., 0., 0., 0., 0., 0., 0., 0.],
        ref_factor=[0., 0., 0., 0., 0., 0.0234, 0.0438, 0.0609, 0.0750, 0.0859, 0.0938, 0.0984,
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ),
    "overflowed_warmup_range": dict(
        params=SchedulerParams(power=4,
                               warmup_start_epoch=2, warmup_end_epoch=8,
                               init_importance_threshold=0, final_importance_threshold=5,
                               importance_regularization_factor=10, steps_per_epoch=3,
                               enable_structured_masking=False),
        ref_threshold=[0., 0., 0., 0., 0., 0., 0., 1.0219, 1.8785, 2.5887, 3.1702, 3.6396, 4.0123, 4.3027, 4.5237],
        ref_factor=[0., 0., 0., 0., 0., 0., 0., 2.0438, 3.7570, 5.1775, 6.3405, 7.2793, 8.0247, 8.6053, 9.0474]
    ),
    "no_warmup": dict(
        params=SchedulerParams(power=4,
                               warmup_start_epoch=-1, warmup_end_epoch=0,
                               init_importance_threshold=-1, final_importance_threshold=2,
                               importance_regularization_factor=5, steps_per_epoch=2,
                               enable_structured_masking=True),
        ref_threshold=[2.] * 10,
        ref_factor=[5.] * 10),
}


@pytest.mark.parametrize('desc',
                         desc_test_decayed_importance_threshold_and_regularization_factor.values(),
                         ids=desc_test_decayed_importance_threshold_and_regularization_factor.keys())
def test_scheduler_decayed_importance_threshold_and_regularization_factor(desc):
    scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=desc['params'].__dict__)
    threshold, factor = [], []
    for epoch in range(5):
        scheduler.epoch_step()
        for batch in range(desc['params'].steps_per_epoch):
            scheduler.step()
            threshold.append(scheduler.current_importance_threshold)
            factor.append(scheduler.current_importance_lambda)
    assert np.allclose(threshold, desc['ref_threshold'], atol=1e-4)
    assert np.allclose(factor, desc['ref_factor'], atol=1e-4)


@pytest.mark.parametrize('desc',
                         desc_test_decayed_importance_threshold_and_regularization_factor.values(),
                         ids=desc_test_decayed_importance_threshold_and_regularization_factor.keys())
def test_scheduler_update_operand_importance_threshold(desc):
    num_minfo = 2
    minfo_list = [Mock() for _ in range(num_minfo)]
    scheduler = MovementPolynomialThresholdScheduler(controller=Mock(sparsified_module_info=minfo_list),
                                                     params=desc['params'].__dict__)
    threshold_dict = defaultdict(list)
    for epoch in range(5):
        scheduler.epoch_step()
        for batch in range(desc['params'].steps_per_epoch):
            scheduler.step()
            for i, minfo in enumerate(minfo_list):
                threshold_dict[f'threshold{i}'].append(minfo.operand.importance_threshold)
    for threshold in threshold_dict.values():
        assert np.allclose(threshold, desc['ref_threshold'], atol=1e-4)


@pytest.mark.parametrize('enable_structured_masking', [True, False])
def test_scheduler_enable_structured_masking(enable_structured_masking: bool):
    num_minfo = 2
    controller = Mock(sparsified_module_info=[Mock() for _ in range(num_minfo)])

    def assert_controller_structured_masking_calls(is_called: bool):
        assert_fn = 'assert_called_once' if is_called else 'assert_not_called'
        getattr(controller.reset_independent_structured_mask, assert_fn)()
        getattr(controller.resolve_structured_mask, assert_fn)()
        getattr(controller.populate_structured_mask, assert_fn)()

    def assert_controller_requires_grad_calls(is_called: bool):
        assert_fn = 'assert_called_once' if is_called else 'assert_not_called'
        for minfo in controller.sparsified_module_info:
            getattr(minfo.operand.requires_grad_, assert_fn)()

    params = SchedulerParams(warmup_start_epoch=0, warmup_end_epoch=1, steps_per_epoch=2,
                             enable_structured_masking=enable_structured_masking)
    scheduler = MovementPolynomialThresholdScheduler(controller, params=params.__dict__)
    scheduler.epoch_step()
    scheduler.step()
    scheduler.step()
    scheduler.epoch_step()
    assert_controller_structured_masking_calls(is_called=False)
    assert_controller_requires_grad_calls(is_called=False)
    scheduler.step()
    assert_controller_structured_masking_calls(is_called=enable_structured_masking)  # check called at this step
    assert_controller_requires_grad_calls(is_called=True)
    scheduler.step()
    scheduler.epoch_step()
    scheduler.step()
    scheduler.step()
    assert_controller_structured_masking_calls(is_called=enable_structured_masking)  # check only called once
    assert_controller_requires_grad_calls(is_called=True)


def test_scheduler_get_state():
    params = SchedulerParams()
    scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
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

    ref_scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
    ref_threshold, ref_factor = [], []
    for epoch in range(5):
        ref_scheduler.epoch_step()
        for batch in range(steps_per_epoch):
            ref_scheduler.step()
            if ref_scheduler.current_step > reload_step:
                ref_threshold.append(ref_scheduler.current_importance_threshold)
                ref_factor.append(ref_scheduler.current_importance_lambda)

    # check state dict is loaded
    scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
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
    scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)

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


def test_scheduler_error_on_improper_warmup_start_epoch_value():
    params = SchedulerParams(warmup_start_epoch=0, steps_per_epoch=None)
    with pytest.raises(ValueError):
        _ = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)


def test_scheduler_error_on_wrong_steps_per_epoch_value():
    params = SchedulerParams(warmup_start_epoch=1, steps_per_epoch=2)
    scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
    scheduler.epoch_step()
    for _ in range(3):
        scheduler.step()
    with pytest.raises(Exception, match='Scheduling may be incorrect'):
        scheduler.epoch_step()
