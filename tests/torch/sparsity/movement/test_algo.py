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
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from pytest import approx
from transformers import TrainingArguments
from transformers.trainer_callback import TrainerControl
from transformers.trainer_callback import TrainerState

import nncf
from nncf.api.compression import CompressionStage
from nncf.common.scopes import matches_any
from nncf.common.scopes import should_consider_scope
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.experimental.torch.sparsity.movement.algo import SUPPORTED_NNCF_MODULES
from nncf.experimental.torch.sparsity.movement.algo import ImportanceLoss
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsifier
from nncf.experimental.torch.sparsity.movement.algo import MovementSparsityController
from nncf.experimental.torch.sparsity.movement.algo import SparseStructure
from nncf.experimental.torch.sparsity.movement.layers import SparseConfig
from nncf.experimental.torch.sparsity.movement.layers import SparseConfigByScope
from nncf.experimental.torch.sparsity.movement.scheduler import MovementPolynomialThresholdScheduler
from nncf.experimental.torch.sparsity.movement.scheduler import MovementSchedulerParams
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskHandler
from nncf.torch import create_compressed_model
from nncf.torch.layer_utils import CompressionParameter
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateWeightAndBias
from tests.torch.sparsity.movement.helpers import FACTOR_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers import LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers import MODEL_SPARSITY_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers import THRESHOLD_NAME_IN_MOVEMENT_STAT
from tests.torch.sparsity.movement.helpers import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from tests.torch.sparsity.movement.helpers import CompressionCallback
from tests.torch.sparsity.movement.helpers import Conv2dPlusLinearRunRecipe
from tests.torch.sparsity.movement.helpers import Conv2dRunRecipe
from tests.torch.sparsity.movement.helpers import DictInTransformerBlockOrder
from tests.torch.sparsity.movement.helpers import LinearRunRecipe
from tests.torch.sparsity.movement.helpers import SwinRunRecipe
from tests.torch.sparsity.movement.helpers import Wav2Vec2RunRecipe
from tests.torch.sparsity.movement.helpers import build_compression_trainer
from tests.torch.sparsity.movement.helpers import force_update_sparsifier_binary_masks_by_threshold
from tests.torch.sparsity.movement.helpers import initialize_sparsifier_parameters_by_linspace
from tests.torch.sparsity.movement.helpers import is_roughly_non_decreasing
from tests.torch.sparsity.movement.helpers import is_roughly_of_same_value

desc_sparse_structures = {
    "explicit_mixed": [
        {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}attention"},
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}Intermediate"},
        {"mode": "per_dim", "axis": 1, "target_scopes": "{re}(?<!Self)Output"},
    ],
    "implicit_all_fine": [],
    "explicit_all_fine": [{"mode": "fine", "sparse_factors": [1, 1], "target_scopes": "{re}.*"}],
    "all_block": [{"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}.*"}],
    "all_per_row": [{"mode": "per_dim", "axis": 0, "target_scopes": "{re}.*"}],
    "all_per_col": [{"mode": "per_dim", "axis": 1, "target_scopes": "{re}.*"}],
    "mixed_of_explicit_block_and_implicit_fine": [
        {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}.*query.*"}
    ],
    "mixed_of_explicit_per_dim_and_implicit_fine": [
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}Intermediate.*"}
    ],
    "mixed_of_explicit_block_or_per_dim_and_implicit_fine": [
        {"mode": "per_dim", "axis": 0, "target_scopes": "{re}Intermediate.*"},
        {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}.*attention.*"},
    ],
    "block_with_target_scopes_list": [
        {"mode": "block", "sparse_factors": [2, 2], "target_scopes": ["{re}query", "{re}key"]}
    ],
}


desc_improper_sparse_structures = {
    "block_not_divisible": dict(
        sparse_structure_by_scopes=[{"mode": "block", "sparse_factors": [3, 3], "target_scopes": "{re}attention"}],
        error=AssertionError,
        match="not a factor of dim axis",
    ),
    "per_dim_wrong_axis": dict(
        sparse_structure_by_scopes=[
            {"mode": "per_dim", "axis": 2, "target_scopes": "{re}attention"},
        ],
        error=ValueError,
        match="Invalid axis id",
    ),
    "duplicate_matches_conflict_config": dict(
        sparse_structure_by_scopes=[
            {"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}attention"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}query"},
        ],
        error=nncf.InternalError,
        match="matched by multiple",
    ),
    "duplicate_matches_same_config": dict(
        sparse_structure_by_scopes=[
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}attention"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}query"},
        ],
        error=nncf.InternalError,
        match="matched by multiple",
    ),
}


@pytest.fixture(scope="function", autouse=True)
def safe_deterministic_state(_safe_deterministic_state):
    pass


class TestControllerCreation:
    @pytest.mark.parametrize(
        "sparse_structure_by_scopes", desc_sparse_structures.values(), ids=desc_sparse_structures.keys()
    )
    @pytest.mark.parametrize(
        "recipe",
        [
            BertRunRecipe().model_config_(hidden_size=4, intermediate_size=6),
            BertRunRecipe().model_config_(hidden_size=4, intermediate_size=6, ffn_bias=False),
            BertRunRecipe()
            .model_config_(hidden_size=4, intermediate_size=6)
            .algo_config_(compression_lr_multiplier=2.0),
            SwinRunRecipe().model_config_(depths=[1, 1], num_heads=[2, 4], mlp_ratio=1.5, qkv_bias=False),
        ],
        ids=["bert", "bert_no_ffn_bias", "bert_with_compression_lr_multiplier", "swin_no_qkv_bias"],
    )
    def test_can_create_movement_sparsity_layers(self, sparse_structure_by_scopes, recipe: BaseMockRunRecipe):
        recipe.algo_config.sparse_structure_by_scopes = sparse_structure_by_scopes
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )
        assert isinstance(compression_ctrl, MovementSparsityController)
        assert isinstance(compression_ctrl.scheduler, MovementPolynomialThresholdScheduler)

        configs = recipe.algo_config.sparse_structure_by_scopes
        compression_lr_multiplier = recipe.algo_config.compression_lr_multiplier
        sparse_configs_by_scopes = [SparseConfigByScope.from_config(c) for c in configs]
        for module, scope in compressed_model.nncf.get_nncf_modules().items():
            if not hasattr(module, "pre_ops"):
                continue
            count_movement_op = 0
            for op in module.pre_ops.values():
                if isinstance(op, UpdateWeightAndBias) and isinstance(op.operand, MovementSparsifier):
                    count_movement_op += 1
                    sparse_config = SparseConfig(SparseStructure.FINE, (1, 1))
                    for sparse_config_by_scope in sparse_configs_by_scopes:
                        if matches_any(str(scope), sparse_config_by_scope.target_scopes):
                            sparse_config = sparse_config_by_scope.sparse_config
                            break
                    self._check_sparsified_layer_mode(op.operand, module, sparse_config, compression_lr_multiplier)
            if should_consider_scope(str(scope), recipe.algo_config.ignored_scopes) and isinstance(
                module, tuple(SUPPORTED_NNCF_MODULES)
            ):
                assert count_movement_op == 1
            else:
                assert count_movement_op == 0

    def _check_sparsified_layer_mode(
        self,
        sparsifier: MovementSparsifier,
        module: NNCFLinear,
        config: SparseConfig,
        compression_lr_multiplier: Optional[float],
    ):
        weight_shape = module.weight.shape
        assert isinstance(sparsifier.weight_importance, CompressionParameter)
        if config.mode == SparseStructure.BLOCK:
            ref_weight_shape = [
                weight_shape[0] // config.sparse_factors[0],
                weight_shape[1] // config.sparse_factors[1],
            ]
        elif config.mode == SparseStructure.PER_DIM:
            ref_weight_shape = [1, weight_shape[1]] if config.sparse_axis == 0 else [weight_shape[0], 1]
        else:
            ref_weight_shape = weight_shape
        ref_weight_importance = torch.zeros(ref_weight_shape)
        assert torch.allclose(sparsifier.weight_importance, ref_weight_importance)
        self._check_tensor_compression_lr_multiplier_hook(sparsifier.weight_importance, compression_lr_multiplier)

        if module.bias is not None:
            assert isinstance(sparsifier.bias_importance, CompressionParameter)
            ref_bias_importance = torch.zeros([ref_weight_importance.shape[0]])
            assert torch.allclose(sparsifier.bias_importance, ref_bias_importance)
            self._check_tensor_compression_lr_multiplier_hook(sparsifier.bias_importance, compression_lr_multiplier)

    def _check_tensor_compression_lr_multiplier_hook(
        self, tensor: torch.Tensor, compression_lr_multiplier: Optional[float]
    ):
        requires_grad = tensor.requires_grad
        tensor.requires_grad_(True)
        tensor.grad = None
        tensor.backward(torch.ones_like(tensor))
        ref_compression_lr_multiplier = 1.0 if compression_lr_multiplier is None else compression_lr_multiplier
        ref_grad = torch.ones_like(tensor) * ref_compression_lr_multiplier
        assert torch.allclose(tensor.grad, ref_grad)
        tensor.grad = None
        tensor.requires_grad_(requires_grad)

    @pytest.mark.parametrize(
        "desc", desc_improper_sparse_structures.values(), ids=desc_improper_sparse_structures.keys()
    )
    def test_error_on_wrong_sparse_structure_by_scopes(self, desc: dict):
        recipe = BertRunRecipe()
        recipe.algo_config_(sparse_structure_by_scopes=desc["sparse_structure_by_scopes"])
        with pytest.raises(desc["error"], match=desc["match"]):
            create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)

    @pytest.mark.parametrize(
        "recipe", [Conv2dRunRecipe(), LinearRunRecipe().algo_config_(ignored_scopes=["{re}model"])]
    )
    def test_error_on_no_supported_layers(self, recipe: BaseMockRunRecipe):
        with pytest.raises(nncf.InternalError, match="No sparsifiable layer"):
            create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)

    @pytest.mark.parametrize("enable_structured_masking", [True, False])
    @pytest.mark.parametrize("run_recipe", [BertRunRecipe(), Wav2Vec2RunRecipe(), SwinRunRecipe(), LinearRunRecipe()])
    def test_can_create_structured_mask_handler_if_supported(
        self, enable_structured_masking: bool, run_recipe: BaseMockRunRecipe
    ):
        recipe = run_recipe.scheduler_params_(enable_structured_masking=enable_structured_masking)
        if enable_structured_masking is True:
            if recipe.supports_structured_masking:
                compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
                assert hasattr(compression_ctrl, "_structured_mask_handler")
                handler = getattr(compression_ctrl, "_structured_mask_handler")
                assert isinstance(handler, StructuredMaskHandler)
            else:
                with pytest.raises(nncf.UnsupportedModelError, match=r"no supported model"):
                    create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        else:
            compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
            assert (not hasattr(compression_ctrl, "_structured_mask_handler")) or getattr(
                compression_ctrl, "_structured_mask_handler"
            ) is None


class TestControllerStats:
    def test_calculate_sparsity(self):
        recipe = Conv2dPlusLinearRunRecipe()
        model = recipe.model()
        for p in model.parameters():
            torch.nn.init.constant_(p, 1.0)
        conv_numel, linear_numel = 56, 6
        compression_ctrl, _ = create_compressed_model(model, recipe.nncf_config(), dump_graphs=False)
        minfo = compression_ctrl.sparsified_module_info[0]
        initialize_sparsifier_parameters_by_linspace(minfo.operand, -1, 1)
        # initialized importance score for weight: [-1, -0.33, 0.33, 1], bias: [-1, 1]
        for threshold, ref_num_zeros in zip([-2, -0.5, 0, 0.5, 2], [0, 2, 3, 4, 4, 6]):
            force_update_sparsifier_binary_masks_by_threshold(minfo.operand, threshold)
            stat = compression_ctrl.statistics().movement_sparsity.model_statistics
            assert stat.sparsity_level == ref_num_zeros / (conv_numel + linear_numel)
            assert stat.sparsity_level_for_layers == ref_num_zeros / linear_numel

    @pytest.mark.parametrize(("warmup_start_epoch", "warmup_end_epoch"), [(1, 2), (0, 1), (2, 3), (2, 5)])
    def test_importance_threshold_and_regularization_factor_range(
        self, tmp_path, warmup_start_epoch: int, warmup_end_epoch: int
    ):
        batch_size = 1
        recipe = LinearRunRecipe(log_dir=tmp_path)
        params = MovementSchedulerParams(
            warmup_start_epoch=warmup_start_epoch,
            warmup_end_epoch=warmup_end_epoch,
            importance_regularization_factor=1,
            enable_structured_masking=False,
            init_importance_threshold=-1,
            final_importance_threshold=1,
            steps_per_epoch=2,
        )
        recipe.algo_config.scheduler_params = params
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )
        callback = CompressionCallback(compression_ctrl)
        mock_dataset = recipe.generate_mock_dataset(int(batch_size * params.steps_per_epoch))
        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            train_dataset=mock_dataset,
            callback=callback,
            batch_size=batch_size,
            num_train_epochs=3,
        )
        trainer.train()
        for step, log in callback.get_compression_log().items():
            # step starts from 1
            if step <= int(warmup_start_epoch * params.steps_per_epoch):
                assert log[FACTOR_NAME_IN_MOVEMENT_STAT] == approx(0.0)
                assert log[THRESHOLD_NAME_IN_MOVEMENT_STAT] == -math.inf
            elif step > int(warmup_end_epoch * params.steps_per_epoch):
                assert log[FACTOR_NAME_IN_MOVEMENT_STAT] == approx(params.importance_regularization_factor)
                assert log[THRESHOLD_NAME_IN_MOVEMENT_STAT] == approx(params.final_importance_threshold)
            else:
                atol = 1e-6
                assert 0.0 <= log[FACTOR_NAME_IN_MOVEMENT_STAT] <= params.importance_regularization_factor + atol
                assert (
                    params.init_importance_threshold - atol
                    <= log[THRESHOLD_NAME_IN_MOVEMENT_STAT]
                    <= params.final_importance_threshold + atol
                )

    @pytest.mark.parametrize("enable_structured_masking", [True, False])
    def test_increasing_sparsity_stats_before_warmup_ends(self, tmp_path, enable_structured_masking: bool):
        recipe = BertRunRecipe(log_dir=tmp_path)
        recipe.model_config.hidden_size = 4
        recipe.model_config.intermediate_size = 6
        recipe.algo_config.scheduler_params.enable_structured_masking = enable_structured_masking
        recipe.algo_config.scheduler_params.steps_per_epoch = 5
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )

        assert isinstance(compression_ctrl.statistics().movement_sparsity, MovementSparsityStatistics)
        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            batch_size=4,
            train_dataset=recipe.generate_mock_dataset(20),
            learning_rate=0.10,
        )
        trainer.train()
        log_by_step = trainer.compression_callback.get_compression_log()
        warmup_start_step = int(recipe.scheduler_params.steps_per_epoch * recipe.scheduler_params.warmup_start_epoch)
        warmup_end_step = int(recipe.scheduler_params.steps_per_epoch * recipe.scheduler_params.warmup_end_epoch)

        threshold_stat = [log[THRESHOLD_NAME_IN_MOVEMENT_STAT] for log in log_by_step.values()]
        assert all(np.isneginf(threshold_stat[:warmup_start_step]))
        assert is_roughly_non_decreasing(threshold_stat[warmup_start_step:warmup_end_step], rtol=1e-2)

        for key in [
            FACTOR_NAME_IN_MOVEMENT_STAT,
            LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT,
            MODEL_SPARSITY_NAME_IN_MOVEMENT_STAT,
        ]:
            stat = [log[key] for log in log_by_step.values()]
            assert is_roughly_non_decreasing(stat[:warmup_end_step], rtol=1e-2)

    @pytest.mark.parametrize("enable_structured_masking", [True, False])
    def test_fixed_sparsity_stats_after_warmup_ends(self, tmp_path, enable_structured_masking: bool):
        recipe = BertRunRecipe(log_dir=tmp_path)
        recipe.model_config_(hidden_size=4, intermediate_size=6)
        recipe.scheduler_params_(enable_structured_masking=enable_structured_masking, steps_per_epoch=5)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )
        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            batch_size=4,
            train_dataset=recipe.generate_mock_dataset(20),
            learning_rate=0.10,
        )
        trainer.train()
        log_by_step = trainer.compression_callback.get_compression_log()
        params = recipe.algo_config.scheduler_params
        warmup_end_step = int(params.steps_per_epoch * params.warmup_end_epoch)

        for key in [LINEAR_LAYER_SPARSITY_NAME_IN_MOVEMENT_STAT, MODEL_SPARSITY_NAME_IN_MOVEMENT_STAT]:
            stat = [log[key] for log in log_by_step.values()]
            if enable_structured_masking is True:
                assert stat[warmup_end_step - 1] >= stat[warmup_end_step]
            else:
                assert stat[warmup_end_step - 1] <= stat[warmup_end_step] + 1e-2
            assert is_roughly_of_same_value(stat[warmup_end_step:], atol=1e-7)


class TestControllerCompressionInfo:
    def test_controller_compression_stage(self):
        recipe = LinearRunRecipe().scheduler_params_(warmup_start_epoch=1, warmup_end_epoch=2, steps_per_epoch=None)
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        assert compression_ctrl.compression_stage() is CompressionStage.UNCOMPRESSED
        # epoch 0
        compression_ctrl.scheduler.epoch_step()
        assert compression_ctrl.compression_stage() is CompressionStage.UNCOMPRESSED
        compression_ctrl.scheduler.step()
        assert compression_ctrl.compression_stage() is CompressionStage.UNCOMPRESSED
        # epoch 1
        compression_ctrl.scheduler.epoch_step()
        assert compression_ctrl.compression_stage() is CompressionStage.PARTIALLY_COMPRESSED
        compression_ctrl.scheduler.step()
        assert compression_ctrl.compression_stage() is CompressionStage.PARTIALLY_COMPRESSED
        # epoch 2 & 3
        for _ in range(2, 4):
            compression_ctrl.scheduler.epoch_step()
            assert compression_ctrl.compression_stage() is CompressionStage.FULLY_COMPRESSED
            compression_ctrl.scheduler.step()
            assert compression_ctrl.compression_stage() is CompressionStage.FULLY_COMPRESSED

    def test_controller_compression_ratio(self, mocker):
        recipe = LinearRunRecipe()
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        mock_stat = NNCFStatistics()
        mock_stat.register("movement_sparsity", MovementSparsityStatistics(mocker.Mock(sparsity_level=0.6), 1, 1))
        stat = mocker.patch.object(compression_ctrl, "statistics", mocker.Mock(return_value=mock_stat))
        assert stat.is_called_once()
        assert compression_ctrl.compression_rate == 0.6


desc_test_controller_structured_mask_resolution = {
    "prune_1head_1channel": dict(
        unstructured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(
                weight=torch.FloatTensor([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
            mhsa_k=dict(
                weight=torch.FloatTensor([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 0, 0, 0]),
            ),
            mhsa_v=dict(
                weight=torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([0, 1, 0, 0]),
            ),
            mhsa_o=dict(
                weight=torch.FloatTensor([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 1, 1, 0]),
            ),
            ffn_i=dict(
                weight=torch.FloatTensor([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]]), bias=torch.FloatTensor([1, 0, 0])
            ),
            ffn_o=dict(
                weight=torch.FloatTensor([[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
        ),
        ref_structured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(
                weight=torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 1, 0, 0]),
            ),
            mhsa_k=dict(
                weight=torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 1, 0, 0]),
            ),
            mhsa_v=dict(
                weight=torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 1, 0, 0]),
            ),
            mhsa_o=dict(
                weight=torch.FloatTensor([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]),
                bias=torch.FloatTensor([1, 1, 1, 1]),
            ),
            ffn_i=dict(
                weight=torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]), bias=torch.FloatTensor([1, 1, 0])
            ),
            ffn_o=dict(
                weight=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
                bias=torch.FloatTensor([1, 1, 1, 1]),
            ),
        ),
    ),
    "prune_1head_1channel_no_mhsa_qkv_bias": dict(
        unstructured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]), bias=None),
            mhsa_k=dict(weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]), bias=None),
            mhsa_v=dict(weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]), bias=None),
            mhsa_o=dict(
                weight=torch.FloatTensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
                bias=torch.FloatTensor([1, 1, 0, 0]),
            ),
            ffn_i=dict(
                weight=torch.FloatTensor([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]]), bias=torch.FloatTensor([1, 0, 0])
            ),
            ffn_o=dict(
                weight=torch.FloatTensor([[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
        ),
        ref_structured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]), bias=None),
            mhsa_k=dict(weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]), bias=None),
            mhsa_v=dict(weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]), bias=None),
            mhsa_o=dict(
                weight=torch.FloatTensor([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]]),
                bias=torch.FloatTensor([1, 1, 1, 1]),
            ),
            ffn_i=dict(
                weight=torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]), bias=torch.FloatTensor([1, 1, 0])
            ),
            ffn_o=dict(
                weight=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
                bias=torch.FloatTensor([1, 1, 1, 1]),
            ),
        ),
    ),
    "prune_1channel_no_mhsa_o_bias": dict(
        unstructured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(
                weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 0, 0, 0]),
            ),
            mhsa_k=dict(
                weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
            mhsa_v=dict(
                weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
            mhsa_o=dict(weight=torch.FloatTensor([[1, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]]), bias=None),
            ffn_i=dict(
                weight=torch.FloatTensor([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]]), bias=torch.FloatTensor([1, 0, 0])
            ),
            ffn_o=dict(
                weight=torch.FloatTensor([[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
        ),
        ref_structured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            mhsa_k=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            mhsa_v=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            mhsa_o=dict(weight=torch.ones((4, 4)), bias=None),
            ffn_i=dict(
                weight=torch.FloatTensor([[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]), bias=torch.FloatTensor([1, 1, 0])
            ),
            ffn_o=dict(
                weight=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]),
                bias=torch.FloatTensor([1, 1, 1, 1]),
            ),
        ),
    ),
    "prune_none_no_ffn_bias": dict(
        unstructured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(
                weight=torch.FloatTensor([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0]]),
                bias=torch.FloatTensor([1, 0, 0, 0]),
            ),
            mhsa_k=dict(
                weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
            mhsa_v=dict(
                weight=torch.FloatTensor([[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]),
                bias=torch.FloatTensor([0, 0, 0, 0]),
            ),
            mhsa_o=dict(
                weight=torch.FloatTensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 1, 0]]),
                bias=torch.FloatTensor([1, 1, 0, 0]),
            ),
            ffn_i=dict(weight=torch.FloatTensor([[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]]), bias=None),
            ffn_o=dict(weight=torch.FloatTensor([[0, 1, 0], [1, 0, 1], [1, 1, 0], [1, 1, 0]]), bias=None),
        ),
        ref_structured_binary_mask=DictInTransformerBlockOrder(
            mhsa_q=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            mhsa_k=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            mhsa_v=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            mhsa_o=dict(weight=torch.ones((4, 4)), bias=torch.ones(4)),
            ffn_i=dict(weight=torch.ones((3, 4)), bias=None),
            ffn_o=dict(weight=torch.ones((4, 3)), bias=None),
        ),
    ),
}


class TestComponentUpdateInTraining:
    @pytest.mark.parametrize(
        "desc",
        desc_test_controller_structured_mask_resolution.values(),
        ids=desc_test_controller_structured_mask_resolution.keys(),
    )
    def test_controller_structured_mask_resolution(self, tmp_path: Path, desc: dict):
        mhsa_qkv_bias = desc["unstructured_binary_mask"]["mhsa_q"]["bias"] is not None
        mhsa_o_bias = desc["unstructured_binary_mask"]["mhsa_o"]["bias"] is not None
        ffn_bias = desc["unstructured_binary_mask"]["ffn_i"]["bias"] is not None
        recipe = BertRunRecipe(log_dir=tmp_path).model_config_(
            mhsa_qkv_bias=mhsa_qkv_bias, mhsa_o_bias=mhsa_o_bias, ffn_bias=ffn_bias
        )
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )
        compressed_model.train()
        module_dict = recipe.get_nncf_modules_in_transformer_block_order(compressed_model)[0]
        module_vs_operand_map = {minfo.module: minfo.operand for minfo in compression_ctrl.sparsified_module_info}
        for unstructured_binary_mask, module in zip(desc["unstructured_binary_mask"].values(), module_dict.values()):
            operand = module_vs_operand_map[module]
            operand.weight_ctx.binary_mask = unstructured_binary_mask["weight"]
            assert (operand.prune_bias is False and unstructured_binary_mask["bias"] is None) or (
                operand.prune_bias is True and unstructured_binary_mask["bias"] is not None
            )
            if operand.prune_bias:
                operand.bias_ctx.binary_mask = unstructured_binary_mask["bias"]

        compression_ctrl.reset_independent_structured_mask()
        compression_ctrl.resolve_structured_mask()
        compression_ctrl.populate_structured_mask()

        for ref_structured_binary_mask, module in zip(
            desc["ref_structured_binary_mask"].values(), module_dict.values()
        ):
            operand = module_vs_operand_map[module]
            assert torch.allclose(operand.weight_ctx.binary_mask, ref_structured_binary_mask["weight"])
            if operand.prune_bias:
                assert torch.allclose(operand.bias_ctx.binary_mask, ref_structured_binary_mask["bias"])

    def test_importance_score_update(self, tmp_path):
        batch_size = 2
        steps_per_epoch = 2
        recipe = LinearRunRecipe(log_dir=tmp_path).scheduler_params_(steps_per_epoch=steps_per_epoch)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )

        class CheckImportanceScoreCallback(CompressionCallback):
            def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
                super().on_step_end(args, state, control, **kwargs)
                for sparse_module in self.compression_ctrl.sparsified_module_info:
                    sparsifier = sparse_module.operand
                    ref_requires_grad = state.epoch <= recipe.scheduler_params.warmup_end_epoch
                    assert torch.count_nonzero(sparsifier.weight_importance) > 0
                    assert sparsifier.weight_importance.requires_grad is ref_requires_grad
                    if sparsifier.prune_bias is not None:
                        assert torch.count_nonzero(sparsifier.bias_importance) > 0
                        assert sparsifier.bias_importance.requires_grad is ref_requires_grad

        callback = CheckImportanceScoreCallback(compression_ctrl)
        mock_dataset = recipe.generate_mock_dataset(steps_per_epoch * batch_size)
        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            train_dataset=mock_dataset,
            callback=callback,
            batch_size=batch_size,
        )
        trainer.train()

    def test_compression_loss_update(self, tmp_path):
        steps_per_epoch = 2
        recipe = LinearRunRecipe(log_dir=tmp_path).scheduler_params_(steps_per_epoch=steps_per_epoch)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )

        class CheckCompressionLossCallback(CompressionCallback):
            def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
                super().on_step_end(args, state, control, **kwargs)
                assert isinstance(self.compression_ctrl.loss, ImportanceLoss)
                for layer in self.compression_ctrl.loss.operands:
                    assert isinstance(layer, MovementSparsifier)
                loss = self.compression_ctrl.loss()
                assert state.epoch is not None
                assert isinstance(loss, torch.Tensor)
                if (
                    recipe.scheduler_params.warmup_start_epoch < state.epoch
                    and state.epoch <= recipe.scheduler_params.warmup_end_epoch
                    and self.compression_ctrl.scheduler.current_importance_regularization_factor > 0
                ):
                    assert loss.requires_grad is True
                    assert loss > 0.0
                else:
                    assert not loss.is_nonzero()

        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            train_dataset=recipe.generate_mock_dataset(steps_per_epoch),
            callback=CheckCompressionLossCallback(compression_ctrl),
        )
        trainer.train()

    @pytest.mark.parametrize(
        "recipe",
        [
            LinearRunRecipe().model_config_(bias=True),
            LinearRunRecipe().model_config_(bias=False),
            SwinRunRecipe(),
        ],
    )
    def test_binary_mask_update(self, tmp_path, recipe: BaseMockRunRecipe):
        steps_per_epoch = 5
        recipe.scheduler_params.steps_per_epoch = steps_per_epoch
        recipe.log_dir_(tmp_path)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )

        class CheckBinaryMaskCallback(CompressionCallback):
            def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
                super().on_step_end(args, state, control, **kwargs)
                if state.epoch <= recipe.scheduler_params.warmup_start_epoch:
                    for minfo in self.compression_ctrl.sparsified_module_info:
                        weight_mask = minfo.operand.weight_ctx.binary_mask
                        ref_weight_mask = torch.ones_like(minfo.module.weight.data)
                        assert torch.allclose(weight_mask, ref_weight_mask)
                        if minfo.operand.prune_bias:
                            bias_mask = minfo.operand.bias_ctx.binary_mask
                            ref_bias_mask = torch.ones_like(minfo.module.bias.data)
                            assert torch.allclose(bias_mask, ref_bias_mask)
                elif state.epoch >= recipe.scheduler_params.warmup_end_epoch:
                    count_sparse_modules = 0
                    for minfo in self.compression_ctrl.sparsified_module_info:
                        weight_mask = minfo.operand.weight_ctx.binary_mask
                        if torch.mean(weight_mask) < 1:
                            count_sparse_modules += 1
                        if minfo.operand.prune_bias:
                            bias_mask = minfo.operand.bias_ctx.binary_mask
                            if torch.mean(bias_mask) < 1.0:
                                count_sparse_modules += 1
                    assert count_sparse_modules > 0

        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            train_dataset=recipe.generate_mock_dataset(steps_per_epoch),
            callback=CheckBinaryMaskCallback(compression_ctrl),
        )
        trainer.train()

    def test_adaptive_init_importance_threshold_update(self, tmp_path):
        steps_per_epoch = 4
        recipe = LinearRunRecipe(log_dir=tmp_path)
        recipe.model_config_(bias=True)
        recipe.scheduler_params_(init_importance_threshold=None, steps_per_epoch=steps_per_epoch)
        compression_ctrl, compressed_model = create_compressed_model(
            recipe.model(), recipe.nncf_config(), dump_graphs=False
        )

        class CheckInitImportanceThresholdCallback(CompressionCallback):
            def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
                super().on_step_begin(args, state, control, **kwargs)
                if state.global_step < recipe.scheduler_params.warmup_start_epoch * steps_per_epoch:
                    assert self.compression_ctrl.scheduler._init_importance_threshold is None
                else:
                    assert isinstance(self.compression_ctrl.scheduler._init_importance_threshold, float)

        trainer = build_compression_trainer(
            tmp_path,
            compression_ctrl,
            compressed_model,
            train_dataset=recipe.generate_mock_dataset(steps_per_epoch),
            batch_size=1,
            callback=CheckInitImportanceThresholdCallback(compression_ctrl),
        )
        trainer.train()
