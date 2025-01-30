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
import logging
import math
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Optional
from unittest.mock import MagicMock
from unittest.mock import Mock

import numpy as np
import pytest
from pytest import approx

from nncf.common.logging import nncf_logger
from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerParams
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerStage
from nncf.torch.model_creation import create_compressed_model
from tests.torch.sparsity.movement.helpers import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from tests.torch.sparsity.movement.helpers import LinearRunRecipe
from tests.torch.sparsity.movement.helpers import SwinRunRecipe
from tests.torch.sparsity.movement.helpers import force_update_sparsifier_binary_masks_by_threshold
from tests.torch.sparsity.movement.helpers import initialize_sparsifier_parameters_by_linspace


class TestSchedulerParams:
    def test_minimal_valid_params(self):
        config = {"warmup_start_epoch": 1, "warmup_end_epoch": 2, "importance_regularization_factor": 1.0}
        params = MovementSchedulerParams.from_dict(config)
        ref_params = MovementSchedulerParams(
            power=3,
            warmup_start_epoch=1,
            warmup_end_epoch=2,
            init_importance_threshold=None,
            final_importance_threshold=0.0,
            importance_regularization_factor=1.0,
            steps_per_epoch=None,
            enable_structured_masking=True,
        )
        assert params.__dict__ == ref_params.__dict__

    @pytest.mark.parametrize('desc', [
        dict(params={},
             error=ValueError,
             match='required in config'),
        dict(params=dict(warmup_start_epoch=1),
             error=ValueError,
             match='required in config'),
        dict(params=dict(warmup_start_epoch=1, warmup_end_epoch=2),
             error=ValueError,
             match='required in config'),
        dict(params=dict(warmup_start_epoch=0, warmup_end_epoch=2,
                         importance_regularization_factor=1, steps_per_epoch=None),
             error=ValueError,
             match='must be >= 1 to enable the auto calculation'),
        dict(params=dict(warmup_start_epoch=3, warmup_end_epoch=2, importance_regularization_factor=1),
             error=ValueError,
             match='0 <= warmup_start_epoch < warmup_end_epoch'),
        dict(params=dict(warmup_start_epoch=-1, warmup_end_epoch=2,
                         importance_regularization_factor=1, steps_per_epoch=4),
             error=ValueError,
             match='0 <= warmup_start_epoch < warmup_end_epoch'),
        dict(params=dict(warmup_start_epoch=1, warmup_end_epoch=2,
                         importance_regularization_factor=-1),
             error=ValueError,
             match='should not be a negative number'),
    ])  # fmt: skip
    def test_error_on_wrong_config(self, desc: dict):
        with pytest.raises(desc["error"], match=desc["match"]):
            _ = MovementSchedulerParams.from_dict(desc["params"])

    @pytest.mark.parametrize(
        "desc",
        [
            dict(
                params=dict(
                    warmup_start_epoch=1,
                    warmup_end_epoch=2,
                    init_importance_threshold=2.0,
                    final_importance_threshold=1.0,
                    importance_regularization_factor=1,
                ),
                match="`init_importance_threshold` is equal to or greater",
            ),
            dict(
                params=dict(
                    warmup_start_epoch=1,
                    warmup_end_epoch=2,
                    init_importance_threshold=2.0,
                    final_importance_threshold=2.0,
                    importance_regularization_factor=1,
                ),
                match="`init_importance_threshold` is equal to or greater",
            ),
        ],
    )
    def test_warn_on_improper_config(self, desc: dict, nncf_caplog):
        with nncf_caplog.at_level(logging.WARNING, logger=nncf_logger.name):
            _ = MovementSchedulerParams.from_dict(desc["params"])
        assert desc["match"] in nncf_caplog.text


desc_current_importance_threshold_and_regularization_factor = {
    'normal_warmup_range': dict(
        params=MovementSchedulerParams(power=2,
                                       warmup_start_epoch=1, warmup_end_epoch=3,
                                       init_importance_threshold=-1, final_importance_threshold=0,
                                       importance_regularization_factor=0.1, steps_per_epoch=4,
                                       enable_structured_masking=False),
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -1., -0.7656, -0.5625, -0.3906, -0.2500, -0.1406,
                       -0.0625, -0.0156, 0., 0., 0., 0., 0., 0., 0., 0.],
        ref_factor=[0., 0., 0., 0., 0., 0.0234, 0.0438, 0.0609, 0.0750, 0.0859, 0.0938, 0.0984,
                    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    ),
    'overflowed_warmup_range': dict(
        params=MovementSchedulerParams(power=4,
                                       warmup_start_epoch=2, warmup_end_epoch=8,
                                       init_importance_threshold=0, final_importance_threshold=5,
                                       importance_regularization_factor=10, steps_per_epoch=3,
                                       enable_structured_masking=False),
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 0., 1.0219, 1.8785,
                       2.5887, 3.1702, 3.6396, 4.0123, 4.3027, 4.5237],
        ref_factor=[0., 0., 0., 0., 0., 0., 0., 2.0438, 3.7570, 5.1775, 6.3405, 7.2793, 8.0247, 8.6053, 9.0474]
    ),
    'unfavored_importance_threshold_and_factor': dict(
        params=MovementSchedulerParams(power=4,
                                       warmup_start_epoch=2, warmup_end_epoch=8,
                                       init_importance_threshold=0, final_importance_threshold=5,
                                       importance_regularization_factor=10, steps_per_epoch=3,
                                       enable_structured_masking=False),
        ref_threshold=[-math.inf, -math.inf, -math.inf, -math.inf, -math.inf, -math.inf, 0., 1.0219, 1.8785,
                       2.5887, 3.1702, 3.6396, 4.0123, 4.3027, 4.5237],
        ref_factor=[0., 0., 0., 0., 0., 0., 0., 2.0438, 3.7570, 5.1775, 6.3405, 7.2793, 8.0247, 8.6053, 9.0474]
    ),
    'unspecified_init_threshold': dict(
        params=MovementSchedulerParams(power=2,
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
}  # fmt: skip


class TestSchedulerStatus:
    @pytest.mark.parametrize("steps_per_epoch", [None, 2])
    def test_current_stage(self, steps_per_epoch: Optional[int]):
        params = MovementSchedulerParams(
            warmup_start_epoch=1,
            warmup_end_epoch=3,
            importance_regularization_factor=1,
            init_importance_threshold=-0.1,
            steps_per_epoch=steps_per_epoch,
        )
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params)
        for epoch in range(5):
            if epoch < params.warmup_start_epoch:
                ref_stage = MovementSchedulerStage.PRE_WARMUP
            elif epoch < params.warmup_end_epoch:
                ref_stage = MovementSchedulerStage.IN_WARMUP
            else:
                ref_stage = MovementSchedulerStage.POST_WARMUP
            scheduler.epoch_step()
            for _ in range(2):
                scheduler.step()
                assert scheduler.current_stage == ref_stage

    @pytest.mark.parametrize(
        "desc",
        desc_current_importance_threshold_and_regularization_factor.values(),
        ids=desc_current_importance_threshold_and_regularization_factor.keys(),
    )
    def test_current_importance_threshold_and_regularization_factor(self, desc, mocker):
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=desc["params"])
        if desc["params"].init_importance_threshold is None:
            mocker.patch.object(
                scheduler, "_calc_init_threshold_from_controller", side_effect=[desc["mock_init_importance_threshold"]]
            )
        threshold, factor = [], []
        for _ in range(5):
            scheduler.epoch_step()
            for _ in range(desc["params"].steps_per_epoch):
                scheduler.step()
                threshold.append(scheduler.current_importance_threshold)
                factor.append(scheduler.current_importance_regularization_factor)
        assert np.allclose(threshold, desc["ref_threshold"], atol=1e-4)
        assert np.allclose(factor, desc["ref_factor"], atol=1e-4)

    @pytest.mark.parametrize("steps_per_epoch", [None, 2])
    @pytest.mark.parametrize("adaptive_init_threshold", [True, False])
    def test_get_state(self, steps_per_epoch: Optional[int], adaptive_init_threshold: bool, mocker):
        actual_steps_per_epoch = steps_per_epoch or 2
        actual_init_threshold = -0.1
        warmup_start_epoch = 1
        params = MovementSchedulerParams(
            warmup_start_epoch=warmup_start_epoch,
            warmup_end_epoch=3,
            importance_regularization_factor=1,
            enable_structured_masking=False,
            init_importance_threshold=None if adaptive_init_threshold else actual_init_threshold,
            steps_per_epoch=steps_per_epoch,
        )
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params)
        if adaptive_init_threshold:
            mocker.patch.object(scheduler, "_calc_init_threshold_from_controller", side_effect=[actual_init_threshold])
        ref_init_importance_threshold = params.init_importance_threshold
        assert scheduler.get_state() == {
            "current_epoch": -1,
            "current_step": -1,
            "_init_importance_threshold": ref_init_importance_threshold,
            "_steps_per_epoch": params.steps_per_epoch,
        }
        for epoch in range(4):
            scheduler.epoch_step()
            ref_steps_per_epoch = actual_steps_per_epoch if epoch >= 1 else params.steps_per_epoch
            assert scheduler.get_state() == {
                "current_epoch": epoch,
                "current_step": epoch * actual_steps_per_epoch - 1,
                "_init_importance_threshold": ref_init_importance_threshold,
                "_steps_per_epoch": ref_steps_per_epoch,
            }
            for batch in range(actual_steps_per_epoch):
                scheduler.step()
                if epoch >= warmup_start_epoch:
                    ref_init_importance_threshold = actual_init_threshold
                assert scheduler.get_state() == {
                    "current_epoch": epoch,
                    "current_step": epoch * actual_steps_per_epoch + batch,
                    "_init_importance_threshold": ref_init_importance_threshold,
                    "_steps_per_epoch": ref_steps_per_epoch,
                }

    @pytest.mark.parametrize(
        "steps_per_epoch", [2, 4, 9, None], ids=["epoch3", "epoch1", "epoch0", "epoch0_unknown_steps_per_epoch"]
    )
    @pytest.mark.parametrize("adaptive_init_threshold", [True, False])
    def test_load_state(self, steps_per_epoch: Optional[int], adaptive_init_threshold: bool, mocker):
        actual_steps_per_epoch = steps_per_epoch or 8
        actual_init_threshold = -0.1
        reload_step = 6
        params = MovementSchedulerParams(
            warmup_start_epoch=1,
            warmup_end_epoch=3,
            importance_regularization_factor=1,
            enable_structured_masking=False,
            init_importance_threshold=None if adaptive_init_threshold else actual_init_threshold,
            steps_per_epoch=steps_per_epoch,
        )
        ref_scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params)
        if adaptive_init_threshold:
            mocker.patch.object(
                ref_scheduler, "_calc_init_threshold_from_controller", side_effect=[actual_init_threshold]
            )

        ref_threshold, ref_factor = [], []
        ref_state: Dict[str, Any] = {}
        for _ in range(5):
            ref_scheduler.epoch_step()
            for _ in range(actual_steps_per_epoch):
                ref_scheduler.step()
                if ref_scheduler.current_step == reload_step:
                    ref_state = deepcopy(ref_scheduler.get_state())
                elif ref_scheduler.current_step > reload_step:
                    ref_threshold.append(ref_scheduler.current_importance_threshold)
                    ref_factor.append(ref_scheduler.current_importance_regularization_factor)

        # check state is loaded
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params)
        if adaptive_init_threshold:
            mocker.patch.object(scheduler, "_calc_init_threshold_from_controller", side_effect=[actual_init_threshold])
        scheduler.load_state(ref_state)
        assert scheduler.current_epoch == ref_state["current_epoch"]
        assert scheduler.current_step == ref_state["current_step"]
        assert scheduler._init_importance_threshold == ref_state["_init_importance_threshold"]
        assert scheduler._steps_per_epoch == ref_state["_steps_per_epoch"]

        # check can resume and continue
        threshold, factor = [], []
        for _ in range(actual_steps_per_epoch - (reload_step + 1) % actual_steps_per_epoch):  # rest batch
            scheduler.step()
            threshold.append(scheduler.current_importance_threshold)
            factor.append(scheduler.current_importance_regularization_factor)
        for _ in range(ref_state["current_epoch"] + 1, 5):
            scheduler.epoch_step()
            for _ in range(actual_steps_per_epoch):
                scheduler.step()
                threshold.append(scheduler.current_importance_threshold)
                factor.append(scheduler.current_importance_regularization_factor)
        assert np.allclose(threshold, ref_threshold)
        assert np.allclose(factor, ref_factor)


class TestSchedulerStepAction:
    @pytest.mark.parametrize(
        "desc",
        desc_current_importance_threshold_and_regularization_factor.values(),
        ids=desc_current_importance_threshold_and_regularization_factor.keys(),
    )
    def test_update_operand(self, desc, mocker):
        num_minfo = 2
        minfo_list = [Mock() for _ in range(num_minfo)]
        scheduler = MovementPolynomialThresholdScheduler(
            controller=Mock(sparsified_module_info=minfo_list), params=desc["params"]
        )
        if desc["params"].init_importance_threshold is None:
            mocker.patch.object(
                scheduler, "_calc_init_threshold_from_controller", side_effect=[desc["mock_init_importance_threshold"]]
            )
        threshold_dict = defaultdict(list)
        factor_dict = defaultdict(list)
        for _ in range(5):
            scheduler.epoch_step()
            for _ in range(desc["params"].steps_per_epoch):
                scheduler.step()
                for i, minfo in enumerate(minfo_list):
                    threshold_dict[i].append(minfo.operand.importance_threshold)
                    factor_dict[i].append(minfo.operand.importance_regularization_factor)
        for threshold in threshold_dict.values():
            assert np.allclose(threshold, desc["ref_threshold"], atol=1e-4)
        for factor in factor_dict.values():
            assert np.allclose(factor, desc["ref_factor"], atol=1e-4)

    @pytest.mark.parametrize("enable_structured_masking", [True, False])
    def test_freeze_operand_on_warmup_end(self, enable_structured_masking: bool):
        num_minfo = 2
        controller = Mock(sparsified_module_info=[Mock() for _ in range(num_minfo)])

        def assert_controller_structured_masking_calls(is_called_once: bool):
            assert_fn = "assert_called_once" if is_called_once else "assert_not_called"
            getattr(controller.reset_independent_structured_mask, assert_fn)()
            getattr(controller.resolve_structured_mask, assert_fn)()
            getattr(controller.populate_structured_mask, assert_fn)()

        params = MovementSchedulerParams(
            warmup_start_epoch=0,
            warmup_end_epoch=1,
            importance_regularization_factor=0.1,
            init_importance_threshold=-0.1,
            steps_per_epoch=2,
            enable_structured_masking=enable_structured_masking,
        )
        scheduler = MovementPolynomialThresholdScheduler(controller, params=params)
        scheduler.epoch_step()
        scheduler.step()
        scheduler.step()
        scheduler.epoch_step()
        assert_controller_structured_masking_calls(is_called_once=False)
        controller.freeze.assert_not_called()
        scheduler.step()
        assert_controller_structured_masking_calls(
            is_called_once=enable_structured_masking
        )  # check called at this step
        controller.freeze.assert_called_once()
        scheduler.step()
        scheduler.epoch_step()
        scheduler.step()
        scheduler.step()
        assert_controller_structured_masking_calls(is_called_once=enable_structured_masking)  # check only called once
        controller.freeze.assert_called_once()


class TestSchedulerInferStepsPerEpoch:
    def test_can_infer_steps_per_epoch(self):
        params = MovementSchedulerParams(
            power=2,
            warmup_start_epoch=1,
            warmup_end_epoch=3,
            init_importance_threshold=-1.0,
            final_importance_threshold=0.0,
            enable_structured_masking=False,
            importance_regularization_factor=0.1,
            steps_per_epoch=None,
        )
        threshold_after_6_step_calls = approx(-0.7656, abs=1e-4)
        factor_after_6_step_calls = approx(0.0234, abs=1e-4)
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params)

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
        params = MovementSchedulerParams(
            warmup_start_epoch=1,
            warmup_end_epoch=3,
            init_importance_threshold=-0.1,
            steps_per_epoch=2,
            importance_regularization_factor=1.0,
        )
        scheduler = MovementPolynomialThresholdScheduler(controller=MagicMock(), params=params)
        scheduler.epoch_step()
        for _ in range(3):
            scheduler.step()
        with pytest.raises(Exception, match="Scheduling may be incorrect"):
            scheduler.epoch_step()


class TestSchedulerAdaptiveInitThreshold:
    @pytest.mark.parametrize('desc', [
        dict(recipe=BertRunRecipe(),
             ref_threshold=0.,
             ref_sparsity=0.0541),
        dict(recipe=BertRunRecipe().model_config_(num_hidden_layers=24, num_attention_heads=16,
                                                  intermediate_size=4096, hidden_size=1024),
             ref_threshold=0.2879,
             ref_sparsity=0.0010),
        dict(recipe=SwinRunRecipe().model_config_(image_size=384, patch_size=4, window_size=12,
                                                  embed_dim=192, mlp_ratio=4,
                                                  depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48)),
             ref_threshold=8.3142,
             ref_sparsity=0.0010),
    ], ids=['bert_toy', 'bert_large', 'swin_large'])  # fmt: skip
    def test_set_init_importance_threshold(self, desc: dict):
        recipe: BaseMockRunRecipe = desc["recipe"]
        ref_threshold: float = desc["ref_threshold"]
        ref_sparsity: float = desc["ref_sparsity"]
        recipe.algo_config.scheduler_params = MovementSchedulerParams(
            warmup_start_epoch=0,
            warmup_end_epoch=3,
            steps_per_epoch=10,
            importance_regularization_factor=1.0,
            enable_structured_masking=False,
            init_importance_threshold=None,
            final_importance_threshold=1e3,
        )
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        for i, minfo in enumerate(sorted(compression_ctrl.sparsified_module_info, key=lambda x: x.module_node_name)):
            initialize_sparsifier_parameters_by_linspace(minfo.operand, i * 2, i * 2 + 1)
        scheduler = compression_ctrl.scheduler
        scheduler.epoch_step()
        scheduler.step()
        for minfo in compression_ctrl.sparsified_module_info:
            force_update_sparsifier_binary_masks_by_threshold(minfo.operand)
        init_importance_threshold = scheduler._init_importance_threshold
        assert init_importance_threshold == approx(ref_threshold, abs=1e-4)
        stat = compression_ctrl.statistics().movement_sparsity
        assert stat.model_statistics.sparsity_level_for_layers == approx(ref_sparsity, abs=1e-4)
        assert stat.importance_threshold == approx(ref_threshold, abs=1e-4)

    def test_calc_init_threshold_called_once(self, tmp_path, mocker):
        recipe = BertRunRecipe(log_dir=tmp_path)
        recipe.scheduler_params_(init_importance_threshold=None)
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        func = mocker.patch.object(compression_ctrl.scheduler, "_calc_init_threshold_from_controller", return_value=-1)
        for _ in range(6):
            compression_ctrl.scheduler.epoch_step()
            for _ in range(4):
                compression_ctrl.scheduler.step()
        func.assert_called_once()

    @pytest.mark.parametrize(
        ("target_sparsity", "ref_threshold"), [(0.001, 1.0), (0.5, 500.0), (0.6, 600.0), (0.999, 999.0)]
    )
    def test_calc_init_threshold_correctness(self, target_sparsity: float, ref_threshold: float):
        recipe = LinearRunRecipe().model_config_(input_size=500, bias=False)  # 2x50 shape linear weight
        recipe.scheduler_params_(init_importance_threshold=None, steps_per_epoch=None)
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        for minfo in compression_ctrl.sparsified_module_info:
            initialize_sparsifier_parameters_by_linspace(minfo.operand, 0.0, 999.0)
        scheduler: MovementPolynomialThresholdScheduler = compression_ctrl.scheduler
        threshold = scheduler._calc_init_threshold_from_controller(target_sparsity)
        assert threshold == approx(ref_threshold)
