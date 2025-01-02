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
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock

import pytest
import torch

from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.sparsity.statistics import SparsifiedLayerSummary
from nncf.common.sparsity.statistics import SparsifiedModelStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import create_table
from nncf.experimental.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.experimental.torch.sparsity.movement.layers import SparseConfig
from nncf.experimental.torch.sparsity.movement.layers import SparseConfigByScope
from nncf.experimental.torch.sparsity.movement.layers import SparseStructure
from nncf.experimental.torch.sparsity.movement.loss import ImportanceLoss
from nncf.torch import create_compressed_model
from nncf.torch.layers import NNCFLinear
from tests.torch.sparsity.movement.helpers import LinearRunRecipe
from tests.torch.sparsity.movement.helpers import initialize_sparsifier_parameters_by_linspace
from tests.torch.sparsity.movement.helpers import mock_linear_nncf_node


class TestSparseConfigByScope:
    @pytest.mark.parametrize(
        "config",
        [
            {"target_scopes": "{re}fine"},
            {"mode": "fine", "target_scopes": "{re}fine"},
            {"mode": "fine", "sparse_factors": [1, 1], "target_scopes": "{re}fine"},
            {"mode": "block", "sparse_factors": [32, 32], "target_scopes": "{re}block"},
            {"mode": "per_dim", "axis": 0, "target_scopes": "{re}prune_row"},
            {"mode": "per_dim", "axis": 1, "target_scopes": "prune_column"},
            {"mode": "per_dim", "axis": 1, "target_scopes": ["prune_column", "{re}another_block"]},
        ],
    )
    def test_create_sparse_config_by_scope(self, config: Dict[str, Any]):
        sparse_config_by_scope = SparseConfigByScope.from_config(config)
        assert isinstance(sparse_config_by_scope, SparseConfigByScope)
        ref_target_scopes = config.pop("target_scopes")
        assert sorted(sparse_config_by_scope.target_scopes) == sorted(ref_target_scopes)
        sparse_config = sparse_config_by_scope.sparse_config
        ref_mode = config.pop("mode", "fine")
        if ref_mode == "fine":
            ref_sparse_config = dict(mode=SparseStructure.FINE, sparse_factors=(1, 1), sparse_axis=None)
        elif ref_mode == "block":
            ref_sparse_config = dict(
                mode=SparseStructure.BLOCK, sparse_factors=tuple(config["sparse_factors"]), sparse_axis=None
            )
        else:
            ref_sparse_config = dict(mode=SparseStructure.PER_DIM, sparse_factors=None, sparse_axis=int(config["axis"]))
        assert sparse_config.__dict__ == ref_sparse_config

    @pytest.mark.parametrize(
        "desc",
        [
            dict(
                config={"mode": "fine", "sparse_factors": [2, 2], "target_scopes": "mock"},
                error_info="\\[1, 1\\] or unspecified",
            ),
            dict(config={"mode": "fine", "axis": 0, "target_scopes": "mock"}, error_info="not expect specified `axis`"),
            dict(config={"mode": "block", "target_scopes": "mock"}, error_info="Missing `sparse_factors`"),
            dict(
                config={"mode": "block", "sparse_factors": [2], "target_scopes": "mock"},
                error_info="expects tuple of two",
            ),
            dict(
                config={"mode": "block", "sparse_factors": [2, 2], "axis": 0, "target_scopes": "mock"},
                error_info="not expect specified `axis`",
            ),
            dict(config={"mode": "per_dim", "target_scopes": "mock"}, error_info="Missing `axis`"),
            dict(
                config={"mode": "per_dim", "axis": 0, "sparse_factors": [1, 1], "target_scopes": "mock"},
                error_info="not expect specified `sparse_factors`",
            ),
            dict(config={"mode": "per_dim", "axis": 0}, error_info="Missing `target_scopes`"),
        ],
    )
    def test_error_on_creating_from_wrong_sparse_config_by_scope(self, desc: dict):
        with pytest.raises(ValueError, match=desc["error_info"]):
            _ = SparseConfigByScope.from_config(desc["config"])


desc_test_sparsifier_forward = {
    "block": dict(
        sparse_structure_by_scopes=[{"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}model"}],
        init_weight_importance=torch.FloatTensor([[0, 1], [0, 1]]),
        init_bias_importance=torch.FloatTensor([1, 0]),
        ref_masked_weight=torch.FloatTensor([[0, 0, 2, 3], [0, 0, 6, 7], [0, 0, 10, 11], [0, 0, 14, 15]]),
        ref_masked_bias=torch.FloatTensor([0, 1, 0, 0]),
    ),
    "per_row": dict(
        sparse_structure_by_scopes=[{"mode": "per_dim", "axis": 0, "target_scopes": "{re}model"}],
        init_weight_importance=torch.FloatTensor([[0], [1], [0], [1]]),
        init_bias_importance=torch.FloatTensor([1, 1, 0, 0]),
        ref_masked_weight=torch.FloatTensor([[0, 0, 0, 0], [4, 5, 6, 7], [0, 0, 0, 0], [12, 13, 14, 15]]),
        ref_masked_bias=torch.FloatTensor([0, 1, 0, 0]),
    ),
    "per_column": dict(
        sparse_structure_by_scopes=[{"mode": "per_dim", "axis": 1, "target_scopes": "{re}model"}],
        init_weight_importance=torch.FloatTensor([0, 1, 0, 1]),
        init_bias_importance=torch.FloatTensor([0]),
        ref_masked_weight=torch.FloatTensor([[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11], [0, 13, 0, 15]]),
        ref_masked_bias=torch.FloatTensor([0, 0, 0, 0]),
    ),
    "fine": dict(
        sparse_structure_by_scopes=[{"mode": "fine", "sparse_factors": [1, 1], "target_scopes": "{re}model"}],
        init_weight_importance=torch.FloatTensor([[0, 1, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]),
        init_bias_importance=torch.FloatTensor([0, 0, 0, 1]),
        ref_masked_weight=torch.FloatTensor([[0, 1, 2, 3], [0, 5, 6, 7], [8, 9, 10, 11], [0, 0, 0, 0]]),
        ref_masked_bias=torch.FloatTensor([0, 0, 0, 3]),
    ),
}


class TestSparsifier:
    @pytest.mark.parametrize("desc", desc_test_sparsifier_forward.values(), ids=desc_test_sparsifier_forward.keys())
    def test_sparsifier_forward(self, tmp_path, desc):
        has_bias = desc["init_bias_importance"] is not None
        recipe = LinearRunRecipe(log_dir=tmp_path)
        recipe.algo_config_(sparse_structure_by_scopes=desc["sparse_structure_by_scopes"])
        recipe.model_config_(input_size=4, num_labels=4, bias=has_bias)
        model = recipe.model()
        compression_ctrl, compressed_model = create_compressed_model(model, recipe.nncf_config(), dump_graphs=False)
        compressed_model.train()
        minfo = compression_ctrl.sparsified_module_info[0]
        module, operand = minfo.module, minfo.operand
        model.model.weight.data.copy_(torch.arange(16).reshape(4, 4).float())
        operand.weight_importance.data.copy_(desc["init_weight_importance"])
        if has_bias:
            model.model.bias.data.copy_(torch.arange(4).float())
            operand.bias_importance.data.copy_(desc["init_bias_importance"])
        operand.importance_threshold = 0.5
        masked_weight, masked_bias = operand(module.weight, module.bias)
        assert torch.allclose(masked_weight, desc["ref_masked_weight"])
        assert torch.allclose(masked_bias, desc["ref_masked_bias"])

        compressed_model.eval()
        with torch.no_grad():
            masked_weight, masked_bias = operand(module.weight, module.bias)
            assert torch.allclose(masked_weight, desc["ref_masked_weight"])
            assert torch.allclose(masked_bias, desc["ref_masked_bias"])

        # In eval mode, changes to importance_threshold will not be propagated to masks.
        operand.importance_threshold = 2
        with torch.no_grad():
            masked_weight, masked_bias = operand(module.weight, module.bias)
            assert torch.allclose(masked_weight, desc["ref_masked_weight"])
            assert torch.allclose(masked_bias, desc["ref_masked_bias"])

    @pytest.mark.parametrize(
        "sparse_structure_by_scopes",
        [
            [{"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}model"}],
            [{"mode": "per_dim", "axis": 0, "target_scopes": "{re}model"}],
            [{"mode": "per_dim", "axis": 1, "target_scopes": "{re}model"}],
            [{"mode": "fine", "sparse_factors": [1, 1], "target_scopes": "{re}model"}],
        ],
    )
    @pytest.mark.parametrize("model_bias", [True, False])
    def test_layer_actual_behavior_matches_sparsifier_mask(self, sparse_structure_by_scopes, model_bias: bool):
        recipe = LinearRunRecipe()
        recipe.model_config_(bias=model_bias)
        recipe.algo_config_(sparse_structure_by_scopes=sparse_structure_by_scopes)
        compression_ctrl, _ = create_compressed_model(recipe.model(), recipe.nncf_config(), dump_graphs=False)
        module_info = compression_ctrl.sparsified_module_info[0]
        operand = module_info.operand
        initialize_sparsifier_parameters_by_linspace(operand)
        operand.importance_threshold = 0.0
        ori_weight, ori_bias = module_info.module.weight, module_info.module.bias
        masked_weight, masked_bias = operand(ori_weight, ori_bias)  # sparsifier forward function
        equiv_weight, equiv_bias = self._calc_linear_layer_equiv_weight_bias(module_info.module)
        assert torch.allclose(equiv_weight, masked_weight)
        if module_info.module.bias is not None:
            assert torch.allclose(equiv_bias, masked_bias)
        else:
            assert masked_bias is None
            assert torch.allclose(equiv_bias, torch.zeros_like(equiv_bias))

    def _calc_linear_layer_equiv_weight_bias(self, module: NNCFLinear):
        in_features = module.in_features
        zero_input = torch.zeros((1, in_features))
        eye_input = torch.eye(in_features)
        with torch.no_grad():
            bias = module(zero_input)
            weight = module(eye_input) - bias
        return weight.T, bias

    def test_apply_binary_mask(self):
        operand = MovementSparsifier(mock_linear_nncf_node(2, 2, bias=True))
        operand.weight_ctx.binary_mask = torch.Tensor([[0.0, 1], [0, 1]])
        operand.bias_ctx.binary_mask = torch.Tensor([0.0, 1])
        weight = torch.Tensor([[1.0, 2], [3, 4]])
        masked_weight = operand.apply_binary_mask(weight, is_bias=False)
        ref_masked_weight = torch.Tensor([[0.0, 2], [0, 4]])
        assert torch.allclose(masked_weight, ref_masked_weight)
        bias = torch.Tensor([1.0, 2])
        masked_bias = operand.apply_binary_mask(bias, is_bias=True)
        ref_masked_bias = torch.Tensor([0.0, 2])
        assert torch.allclose(masked_bias, ref_masked_bias)

    @pytest.mark.parametrize("layerwise_loss_lambda", [0.5, 2.0])
    @pytest.mark.parametrize("importance_regularization_factor", [0.0, 1.0])
    @pytest.mark.parametrize("frozen", [True, False])
    @pytest.mark.parametrize(
        "desc",
        [
            dict(
                sparse_cfg=SparseConfig(mode=SparseStructure.FINE),
                weight_importance=torch.FloatTensor([[0, 1], [1, -1]]),
                bias_importance=torch.FloatTensor([-2, 2]),
                raw_loss=1.0578,
            ),
            dict(
                sparse_cfg=SparseConfig(mode=SparseStructure.BLOCK, sparse_factors=(2, 2)),
                weight_importance=torch.FloatTensor([1]),
                bias_importance=torch.FloatTensor([-2]),
                raw_loss=3.1626,
            ),
            dict(
                sparse_cfg=SparseConfig(mode=SparseStructure.PER_DIM, sparse_axis=0),
                weight_importance=torch.FloatTensor([[2], [3]]),
                bias_importance=torch.FloatTensor([1, 4]),
                raw_loss=2.6899,
            ),
            dict(
                sparse_cfg=SparseConfig(mode=SparseStructure.PER_DIM, sparse_axis=1),
                weight_importance=torch.FloatTensor([[2, 3]]),
                bias_importance=None,
                raw_loss=1.8334,
            ),
        ],
    )
    def test_calculate_sparsifier_loss(
        self, layerwise_loss_lambda: float, importance_regularization_factor: float, frozen: bool, desc: dict
    ):
        has_bias = desc["bias_importance"] is not None
        operand = MovementSparsifier(
            mock_linear_nncf_node(2, 2, bias=has_bias),
            sparse_cfg=desc["sparse_cfg"],
            frozen=frozen,
            layerwise_loss_lambda=layerwise_loss_lambda,
        )
        with torch.no_grad():
            operand.weight_importance.copy_(desc["weight_importance"])
            if has_bias:
                operand.bias_importance.copy_(desc["bias_importance"])
        loss = operand.loss()
        assert torch.allclose(loss, torch.zeros(1))

        operand.importance_regularization_factor = importance_regularization_factor
        loss = operand.loss()
        ref_loss = (
            torch.zeros(1)
            if frozen
            else torch.tensor(desc["raw_loss"] * importance_regularization_factor * layerwise_loss_lambda)
        )
        assert torch.allclose(loss, ref_loss, atol=2e-4)

        ref_requires_grad = not frozen and importance_regularization_factor != 0
        assert loss.requires_grad is ref_requires_grad

    def test_requires_grad(self):
        operand = MovementSparsifier(mock_linear_nncf_node(2, 2, bias=True), frozen=False)
        operand.importance_regularization_factor = 1.0
        for set_grad in [False, True]:
            operand.requires_grad_(set_grad)
            assert operand.weight_importance.requires_grad is set_grad
            assert operand.bias_importance.requires_grad is set_grad
            assert operand.frozen is (not set_grad)
            assert operand.loss().requires_grad is set_grad

    @pytest.mark.parametrize(
        ("sparse_cfg", "bias", "ref_weight_importance_shape"),
        [
            (SparseConfig(SparseStructure.FINE), True, (4, 6)),
            (SparseConfig(SparseStructure.BLOCK, (2, 3)), True, (2, 2)),
            (SparseConfig(SparseStructure.PER_DIM, sparse_axis=0), True, (4, 1)),
            (SparseConfig(SparseStructure.PER_DIM, sparse_axis=1), False, (1, 6)),
        ],
    )
    def test_get_importance_shape(
        self, sparse_cfg: SparseConfig, bias: bool, ref_weight_importance_shape: Tuple[int, int]
    ):
        weight_shape = (4, 6)
        operand = MovementSparsifier(
            mock_linear_nncf_node(weight_shape[1], weight_shape[0], bias=bias), sparse_cfg=sparse_cfg, frozen=False
        )
        weight_importance = operand.get_importance(is_bias=False, expanded=False)
        assert weight_importance.shape == torch.Size(ref_weight_importance_shape)
        weight_importance_expanded = operand.get_importance(is_bias=False, expanded=True)
        assert weight_importance_expanded.shape == torch.Size(weight_shape)
        if bias:
            bias_importance = operand.get_importance(is_bias=True, expanded=False)
            assert bias_importance.shape == torch.Size(ref_weight_importance_shape[:1])
            bias_importance_expanded = operand.get_importance(is_bias=True, expanded=True)
            assert bias_importance_expanded.shape == torch.Size(weight_shape[:1])
        else:
            with pytest.raises(ValueError):
                operand.get_importance(is_bias=True)


class TestFunctions:
    @pytest.mark.parametrize(
        ("input_tensor", "threshold", "max_percentile", "ref_output_tensor"),
        [
            (torch.FloatTensor([1, 2, 3, 4]), 0.0, 0.9, torch.FloatTensor([1, 1, 1, 1])),
            (torch.FloatTensor([1, 2, 3, 4]), 2.5, 0.8, torch.FloatTensor([0, 0, 1, 1])),
            (torch.FloatTensor([1, 2, 3, 4]), 2.5, 0.2, torch.FloatTensor([0, 1, 1, 1])),
            (torch.FloatTensor([1, 2, 3, 4]), 5.0, 0.8, torch.FloatTensor([0, 0, 0, 1])),
            (torch.FloatTensor([1, 1, 1, 1]), 5.0, 0.8, torch.FloatTensor([0, 0, 0, 0])),
        ],
    )
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_binary_mask_by_threshold(
        self,
        input_tensor: torch.Tensor,
        threshold: float,
        max_percentile: float,
        ref_output_tensor: torch.Tensor,
        requires_grad: bool,
        use_cuda: bool,
    ):
        if not torch.cuda.is_available() and use_cuda is True:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        device = torch.device("cuda" if use_cuda else "cpu")
        input_tensor = input_tensor.clone().to(device).requires_grad_(requires_grad)
        output_tensor = binary_mask_by_threshold(input_tensor, threshold, max_percentile)
        ref_output_tensor = ref_output_tensor.clone().to(device)
        assert output_tensor.device == input_tensor.device
        assert output_tensor.requires_grad is requires_grad
        assert torch.allclose(output_tensor, ref_output_tensor)
        if requires_grad:
            assert output_tensor.grad_fn.name().startswith("STThresholdBackward")
            output_tensor.sum().backward()
            ref_grad_tensor = torch.ones_like(input_tensor)
            assert torch.allclose(input_tensor.grad, ref_grad_tensor)


class TestImportanceLoss:
    @pytest.mark.parametrize(
        "desc",
        [
            dict(disable=False, sparse_layers_loss=(1.0, 2.0, 3.0), ref_output=2.0),
            dict(disable=True, sparse_layers_loss=(1.0,), ref_output=1.0),
        ],
    )
    @pytest.mark.parametrize("requires_grad", [True, False])
    def test_importance_loss_forward(self, desc, requires_grad: bool, use_cuda: bool):
        if (not torch.cuda.is_available()) and use_cuda:
            pytest.skip("Skipping CUDA test cases for CPU only setups")
        device = torch.device("cuda" if use_cuda else "cpu")
        operands = []
        for loss_val in desc["sparse_layers_loss"]:
            operand = MovementSparsifier(mock_linear_nncf_node(), frozen=False)
            if use_cuda:
                operand = operand.cuda()
            loss_tensor = torch.tensor(loss_val, requires_grad=requires_grad, device=device)
            operand.loss = MagicMock(return_value=loss_tensor)
            operands.append(operand)

        loss_module = ImportanceLoss(operands)
        if desc["disable"]:
            loss_module.disable()
        output = loss_module()
        assert isinstance(output, torch.Tensor) and output.device.type == device.type
        if desc["disable"]:
            assert output.requires_grad is False
            assert torch.allclose(output, torch.zeros_like(output))
        else:
            for operand in operands:
                operand.loss.assert_called_once()
            assert output.requires_grad is requires_grad
            assert torch.allclose(output, torch.tensor(desc["ref_output"]))

    @pytest.mark.cuda
    def test_importance_loss_adapts_to_device_change(self):
        if not torch.cuda.is_available():
            pytest.skip("requires GPU")
        sparsifier = MovementSparsifier(mock_linear_nncf_node(), frozen=False)
        loss_module = ImportanceLoss([sparsifier])
        loss_cpu = loss_module()
        assert loss_cpu.device.type == "cpu"
        sparsifier.cuda()
        loss_cuda = loss_module()
        assert loss_cuda.device.type == "cuda"


class TestMovementSparsityStatistics:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.importance_threshold = 1.0
        self.importance_regularization_factor = 2.0
        summary = SparsifiedLayerSummary("layer", [1, 1], 0.5, 0.5)
        model_stats = SparsifiedModelStatistics(0.25, 0.5, [summary])
        movement_stats = MovementSparsityStatistics(
            model_stats, self.importance_threshold, self.importance_regularization_factor
        )
        self.movement_stats = movement_stats

    def test_movement_sparsity_statistics_string(self):
        movement_stats = self.movement_stats
        output_str = movement_stats.to_str()
        assert movement_stats.to_str() in output_str
        assert (
            create_table(
                header=["Statistic's name", "Value"],
                rows=[
                    ["Mask Importance Threshold", self.importance_threshold],
                    ["Importance Regularization Factor", self.importance_regularization_factor],
                ],
            )
            in output_str
        )

    def test_nncf_stats_can_register_movement_sparsity_stats(self):
        movement_stats = self.movement_stats
        nncf_stats = NNCFStatistics()
        nncf_stats.register("movement_sparsity", movement_stats)
        assert hasattr(nncf_stats, "movement_sparsity")
        assert nncf_stats.movement_sparsity == movement_stats
