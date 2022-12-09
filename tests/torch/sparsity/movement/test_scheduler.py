from collections import defaultdict
import logging
import math
from unittest.mock import MagicMock
from unittest.mock import Mock

import numpy as np
import pytest
from pytest import approx

from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerStage
from nncf.torch.model_creation import create_compressed_model
from tests.torch.sparsity.movement.helpers import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from tests.torch.sparsity.movement.helpers import LinearRunRecipe
from tests.torch.sparsity.movement.helpers import SchedulerParams
from tests.torch.sparsity.movement.helpers import SwinRunRecipe
from tests.torch.sparsity.movement.helpers import initialize_sparsifier_parameters_by_linspace


class TestSchedulerCreation:
    @pytest.mark.parametrize('desc', [
        dict(params=SchedulerParams(warmup_start_epoch=0, steps_per_epoch=None),
             error=ValueError,
             match='must be >= 1 to enable the auto calculation'),
        dict(params=SchedulerParams(warmup_start_epoch=-1),
             error=ValueError,
             match='0 <= warmup_start_epoch < warmup_end_epoch'),
        dict(params=SchedulerParams(warmup_start_epoch=1, warmup_end_epoch=1),
             error=ValueError,
             match='0 <= warmup_start_epoch < warmup_end_epoch'),
        dict(params=SchedulerParams(importance_regularization_factor=-1),
             error=ValueError,
             match='should not be a negative number'),
    ])
    def test_error_on_wrong_config(self, desc: dict):
        with pytest.raises(desc['error'], match=desc['match']):
            _ = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=desc['params'].__dict__)

    @pytest.mark.parametrize('desc', [
        dict(params=SchedulerParams(init_importance_threshold=2., final_importance_threshold=1.),
             match='`init_importance_threshold` is equal to or greater'),
        dict(params=SchedulerParams(init_importance_threshold=2., final_importance_threshold=2.),
             match='`init_importance_threshold` is equal to or greater'),
    ])
    def test_warn_on_improper_config(self, desc: dict, mocker, caplog):
        with caplog.at_level(logging.WARNING, logger='nncf'):
            mocker.patch.object(logging.getLogger('nncf'), 'propagate', True)
            _ = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=desc['params'].__dict__)
        assert desc['match'] in caplog.text


desc_current_importance_threshold_and_regularization_factor = {
    "normal_warmup_range": dict(
        params=SchedulerParams(power=2,
                               warmup_start_epoch=1, warmup_end_epoch=3,
                               init_importance_threshold=-1, final_importance_threshold=0,
                               importance_regularization_factor=0.1, steps_per_epoch=4,
                               enable_structured_masking=False),
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -1., -0.7656, -0.5625, -0.3906, -0.2500, -0.1406,
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
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 0., 1.0219, 1.8785,
                       2.5887, 3.1702, 3.6396, 4.0123, 4.3027, 4.5237],
        ref_factor=[0., 0., 0., 0., 0., 0., 0., 2.0438, 3.7570, 5.1775, 6.3405, 7.2793, 8.0247, 8.6053, 9.0474]
    ),
    "unfavored_importance_threshold_and_factor": dict(
        params=SchedulerParams(power=4,
                               warmup_start_epoch=2, warmup_end_epoch=8,
                               init_importance_threshold=0, final_importance_threshold=5,
                               importance_regularization_factor=10, steps_per_epoch=3,
                               enable_structured_masking=False),
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 0., 1.0219, 1.8785,
                       2.5887, 3.1702, 3.6396, 4.0123, 4.3027, 4.5237],
        ref_factor=[0., 0., 0., 0., 0., 0., 0., 2.0438, 3.7570, 5.1775, 6.3405, 7.2793, 8.0247, 8.6053, 9.0474]
    ),
    "unspecified_init_threshold": dict(
        params=SchedulerParams(power=2,
                               warmup_start_epoch=1, warmup_end_epoch=3,
                               init_importance_threshold=None, final_importance_threshold=0,
                               importance_regularization_factor=0.1, steps_per_epoch=4,
                               enable_structured_masking=False),
        mock_init_importance_threshold=-1.,
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -1., -0.7656, -0.5625, -0.3906, -0.2500, -0.1406,
                       -0.0625, -0.0156, 0., 0., 0., 0., 0., 0., 0., 0.],
        ref_factor=[0., 0., 0., 0., 0., 0.0234, 0.0438, 0.0609, 0.0750, 0.0859, 0.0938, 0.0984,
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    )
}


class TestSchedulerStatus:
    @pytest.mark.parametrize('params', [
        SchedulerParams(steps_per_epoch=None),
        SchedulerParams(steps_per_epoch=2)
    ])
    def test_current_stage(self, params: SchedulerParams):
        params.warmup_start_epoch = 1
        params.warmup_end_epoch = 3
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
        for epoch in range(5):
            if epoch < 1:
                ref_stage = MovementSchedulerStage.PRE_WARMUP
            elif epoch < 3:
                ref_stage = MovementSchedulerStage.IN_WARMUP
            else:
                ref_stage = MovementSchedulerStage.POST_WARMUP
            scheduler.epoch_step()
            for _ in range(2):
                scheduler.step()
                assert scheduler.current_stage == ref_stage

    @pytest.mark.parametrize('desc',
                             desc_current_importance_threshold_and_regularization_factor.values(),
                             ids=desc_current_importance_threshold_and_regularization_factor.keys())
    def test_current_importance_threshold_and_regularization_factor(self, desc, mocker):
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=desc['params'].__dict__)
        if desc['params'].init_importance_threshold is None:
            mocker.patch.object(scheduler, '_calc_init_threshold_from_controller',
                                return_value=desc['mock_init_importance_threshold'])
        threshold, factor = [], []
        for _ in range(5):
            scheduler.epoch_step()
            for _ in range(desc['params'].steps_per_epoch):
                scheduler.step()
                threshold.append(scheduler.current_importance_threshold)
                factor.append(scheduler.current_importance_regularization_factor)
        assert np.allclose(threshold, desc['ref_threshold'], atol=1e-4)
        assert np.allclose(factor, desc['ref_factor'], atol=1e-4)

    def test_get_state(self):
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
        SchedulerParams(steps_per_epoch=9),
        SchedulerParams(steps_per_epoch=None),
    ], ids=['epoch3', 'epoch1', 'epoch0', 'epoch0_unknown_steps_per_epoch'])
    def test_load_state(self, params):
        reload_step = 6
        steps_per_epoch = params.steps_per_epoch or 8

        ref_scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
        ref_threshold, ref_factor = [], []
        for _ in range(5):
            ref_scheduler.epoch_step()
            for _ in range(steps_per_epoch):
                ref_scheduler.step()
                if ref_scheduler.current_step > reload_step:
                    ref_threshold.append(ref_scheduler.current_importance_threshold)
                    ref_factor.append(ref_scheduler.current_importance_regularization_factor)

        # check state dict is loaded
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
        ref_state = {'current_epoch': reload_step // steps_per_epoch,
                     'current_step': reload_step,
                     '_steps_per_epoch': params.steps_per_epoch}
        scheduler.load_state(ref_state)
        assert scheduler.current_epoch == ref_state['current_epoch']
        assert scheduler.current_step == ref_state['current_step']
        assert scheduler._steps_per_epoch == ref_state['_steps_per_epoch']  # pylint: disable=protected-access

        # check can resume and continue
        threshold, factor = [], []
        for _ in range(steps_per_epoch - (reload_step + 1) % steps_per_epoch):  # rest batch
            scheduler.step()
            threshold.append(scheduler.current_importance_threshold)
            factor.append(scheduler.current_importance_regularization_factor)
        for _ in range(ref_state['current_epoch'] + 1, 5):
            scheduler.epoch_step()
            for _ in range(steps_per_epoch):
                scheduler.step()
                threshold.append(scheduler.current_importance_threshold)
                factor.append(scheduler.current_importance_regularization_factor)
        assert np.allclose(threshold, ref_threshold)
        assert np.allclose(factor, ref_factor)


class TestSchedulerStepAction:
    @pytest.mark.parametrize('desc', desc_current_importance_threshold_and_regularization_factor.values(),
                             ids=desc_current_importance_threshold_and_regularization_factor.keys())
    def test_update_operand(self, desc, mocker):
        num_minfo = 2
        minfo_list = [Mock() for _ in range(num_minfo)]
        scheduler = MovementPolynomialThresholdScheduler(controller=Mock(sparsified_module_info=minfo_list),
                                                         params=desc['params'].__dict__)
        if desc['params'].init_importance_threshold is None:
            mocker.patch.object(scheduler, '_calc_init_threshold_from_controller',
                                return_value=desc['mock_init_importance_threshold'])
        threshold_dict = defaultdict(list)
        factor_dict = defaultdict(list)
        for _ in range(5):
            scheduler.epoch_step()
            for _ in range(desc['params'].steps_per_epoch):
                scheduler.step()
                for i, minfo in enumerate(minfo_list):
                    threshold_dict[i].append(minfo.operand.importance_threshold)
                    factor_dict[i].append(minfo.operand.importance_regularization_factor)
        for threshold in threshold_dict.values():
            assert np.allclose(threshold, desc['ref_threshold'], atol=1e-4)
        for factor in factor_dict.values():
            assert np.allclose(factor, desc['ref_factor'], atol=1e-4)

    @pytest.mark.parametrize('enable_structured_masking', [True, False])
    def test_freeze_operand_on_warmup_end(self, enable_structured_masking: bool):
        num_minfo = 2
        controller = Mock(sparsified_module_info=[Mock() for _ in range(num_minfo)])

        def assert_controller_structured_masking_calls(is_called_once: bool):
            assert_fn = 'assert_called_once' if is_called_once else 'assert_not_called'
            getattr(controller.reset_independent_structured_mask, assert_fn)()
            getattr(controller.resolve_structured_mask, assert_fn)()
            getattr(controller.populate_structured_mask, assert_fn)()

        params = SchedulerParams(warmup_start_epoch=0, warmup_end_epoch=1, steps_per_epoch=2,
                                 enable_structured_masking=enable_structured_masking)
        scheduler = MovementPolynomialThresholdScheduler(controller, params=params.__dict__)
        scheduler.epoch_step()
        scheduler.step()
        scheduler.step()
        scheduler.epoch_step()
        assert_controller_structured_masking_calls(is_called_once=False)
        controller.freeze.assert_not_called()
        scheduler.step()
        assert_controller_structured_masking_calls(
            is_called_once=enable_structured_masking)  # check called at this step
        controller.freeze.assert_called_once()
        scheduler.step()
        scheduler.epoch_step()
        scheduler.step()
        scheduler.step()
        assert_controller_structured_masking_calls(is_called_once=enable_structured_masking)  # check only called once
        controller.freeze.assert_called_once()


class TestSchedulerInferStepsPerEpoch:
    # pylint: disable=protected-access
    def test_can_infer_steps_per_epoch(self):
        params = SchedulerParams(power=2, warmup_start_epoch=1, warmup_end_epoch=3,
                                 init_importance_threshold=-1., final_importance_threshold=0.,
                                 importance_regularization_factor=0.1, steps_per_epoch=None)
        threshold_after_6_step_calls = approx(-0.7656, abs=1e-4)
        factor_after_6_step_calls = approx(0.0234, abs=1e-4)
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)

        scheduler.epoch_step()
        for _ in range(4):
            scheduler.step()
        assert scheduler._steps_per_epoch is None

        scheduler.epoch_step()
        assert scheduler._steps_per_epoch == 4
        scheduler.step()
        scheduler.step()
        assert scheduler.current_importance_threshold == threshold_after_6_step_calls
        assert scheduler.current_importance_regularization_factor == factor_after_6_step_calls

    def test_error_on_wrong_steps_per_epoch_value(self):
        params = SchedulerParams(warmup_start_epoch=1, steps_per_epoch=2)
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params.__dict__)
        scheduler.epoch_step()
        for _ in range(3):
            scheduler.step()
        with pytest.raises(Exception, match='Scheduling may be incorrect'):
            scheduler.epoch_step()


class TestSchedulerAdaptiveInitThreshold:
    @pytest.mark.parametrize(('recipe', 'ref_threshold'), [
        (BertRunRecipe.from_default(), -1),
        (BertRunRecipe.from_default(num_hidden_layers=24, num_attention_heads=16,
                                    intermediate_size=4096, hidden_size=1024
                                    ), -0.99840),
        (SwinRunRecipe.from_default(image_size=384, patch_size=4, window_size=12,
                                    embed_dim=192, mlp_ratio=4,
                                    depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48)
                                    ), -0.99837)
    ], ids=['bert_toy', 'bert_large', 'swin_large'])
    def test_set_init_importance_threshold(self, recipe: BaseMockRunRecipe, ref_threshold: float):
        from nncf.common.utils.logger import set_log_level
        set_log_level(logging.ERROR)
        recipe.set(steps_per_epoch=10, warmup_start_epoch=0,
                   enable_structured_masking=False, init_importance_threshold=None)
        compression_ctrl, _ = create_compressed_model(recipe.model,
                                                      recipe.nncf_config,
                                                      dump_graphs=False)
        for minfo in compression_ctrl.sparsified_module_info:
            initialize_sparsifier_parameters_by_linspace(minfo.operand, -1., 1.)
        scheduler = compression_ctrl.scheduler
        scheduler.epoch_step()
        scheduler.step()
        assert scheduler.init_importance_threshold == approx(ref_threshold, abs=1e-5)
        stat = compression_ctrl.statistics().movement_sparsity
        assert stat.model_statistics.sparsity_level_for_layers == approx(0.001, abs=1e-3)
        assert stat.importance_threshold == approx(ref_threshold, abs=1e-4)

    @pytest.mark.parametrize(('target_sparsity', 'ref_threshold'),
                             [(0.001, 1.), (0.5, 500.), (0.6, 600.), (0.999, 999.)])
    def test_calculate_threshold_value_function(self, target_sparsity: float, ref_threshold: float):
        recipe = LinearRunRecipe.from_default(input_size=500, bias=False,
                                              init_importance_threshold=None,
                                              steps_per_epoch=None)  # 2x50 shape linear weight
        compression_ctrl, _ = create_compressed_model(recipe.model,
                                                      recipe.nncf_config,
                                                      dump_graphs=False)
        for minfo in compression_ctrl.sparsified_module_info:
            initialize_sparsifier_parameters_by_linspace(minfo.operand, 0., 999.)
        scheduler: MovementPolynomialThresholdScheduler = compression_ctrl.scheduler
        threshold = scheduler._calc_init_threshold_from_controller(target_sparsity)  # pylint: disable=protected-access
        assert threshold == approx(ref_threshold)
