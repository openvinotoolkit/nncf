from typing import Any, Dict
from unittest.mock import Mock
from unittest.mock import call

import pytest
from pytest import approx
import torch

from nncf.torch import create_compressed_model
from nncf.common.sparsity.statistics import MovementSparsityStatistics
from nncf.common.sparsity.statistics import SparsifiedLayerSummary
from nncf.common.sparsity.statistics import SparsifiedModelStatistics
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import create_table
from nncf.experimental.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.experimental.torch.sparsity.movement.layers import SparseConfigByScope
from nncf.experimental.torch.sparsity.movement.layers import SparseStructure
from nncf.experimental.torch.sparsity.movement.loss import ImportanceLoss
from tests.torch.sparsity.movement.helpers import LinearRunRecipe
from tests.torch.sparsity.movement.helpers import ensure_tensor

desc_test_sparsifier_forward = {
    "block": dict(
        sparse_structure_by_scopes=[{"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}model"}],
        init_weight_importance=ensure_tensor([[0, 1], [0, 1]]),
        init_bias_importance=ensure_tensor([1, 0]),
        ref_masked_weight=ensure_tensor([[0, 0, 2, 3], [0, 0, 6, 7], [0, 0, 10, 11], [0, 0, 14, 15]]),
        ref_masked_bias=ensure_tensor([0, 1, 0, 0]),
    ),
    "per_row": dict(
        sparse_structure_by_scopes=[{"mode": "per_dim", "axis": 0, "target_scopes": "{re}model"}],
        init_weight_importance=ensure_tensor([[0], [1], [0], [1]]),
        init_bias_importance=ensure_tensor([1, 1, 0, 0]),
        ref_masked_weight=ensure_tensor([[0] * 4, [4, 5, 6, 7], [0] * 4, [12, 13, 14, 15]]),
        ref_masked_bias=ensure_tensor([0, 1, 0, 0]),
    ),
    "per_column": dict(
        sparse_structure_by_scopes=[{"mode": "per_dim", "axis": 1, "target_scopes": "{re}model"}],
        init_weight_importance=ensure_tensor([0, 1, 0, 1]),
        init_bias_importance=ensure_tensor([0]),
        ref_masked_weight=ensure_tensor([[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11], [0, 13, 0, 15]]),
        ref_masked_bias=ensure_tensor([0, 0, 0, 0]),
    ),
    "fine": dict(
        sparse_structure_by_scopes=[{"mode": "fine", "sparse_factors": [1, 1], "target_scopes": "{re}model"}],
        init_weight_importance=ensure_tensor([[0, 1, 1, 1], [0, 1, 1, 1], [1] * 4, [0] * 4]),
        init_bias_importance=ensure_tensor([0, 0, 0, 1]),
        ref_masked_weight=ensure_tensor([[0, 1, 2, 3], [0, 5, 6, 7], [8, 9, 10, 11], [0] * 4]),
        ref_masked_bias=ensure_tensor([0, 0, 0, 3]),
    ),
}


@pytest.mark.parametrize('desc', desc_test_sparsifier_forward.values(),
                         ids=desc_test_sparsifier_forward.keys())
def test_sparsifier_forward(tmp_path, desc):
    has_bias = desc['init_bias_importance'] is not None
    recipe = LinearRunRecipe.from_default(
        input_size=4,
        num_classes=4,
        bias=has_bias,
        sparse_structure_by_scopes=desc['sparse_structure_by_scopes'],
        enable_structured_masking=False,
        log_dir=tmp_path)
    model = recipe.model
    compression_ctrl, compressed_model = create_compressed_model(model,
                                                                 recipe.nncf_config,
                                                                 dump_graphs=False)
    compressed_model.train()
    minfo = compression_ctrl.sparsified_module_info[0]
    module, operand = minfo.module, minfo.operand
    model.model.weight.data.copy_(torch.arange(16).reshape(4, 4).float())
    operand.weight_importance.data.copy_(desc['init_weight_importance'])
    if has_bias:
        model.model.bias.data.copy_(torch.arange(4).float())
        operand.bias_importance.data.copy_(desc['init_bias_importance'])
    operand.importance_threshold = 0.5
    masked_weight, masked_bias = operand(module.weight, module.bias)
    assert torch.allclose(masked_weight, desc['ref_masked_weight'])
    assert torch.allclose(masked_bias, desc['ref_masked_bias'])

    compressed_model.eval()
    with torch.no_grad():
        masked_weight, masked_bias = operand(module.weight, module.bias)
        assert torch.allclose(masked_weight, desc['ref_masked_weight'])
        assert torch.allclose(masked_bias, desc['ref_masked_bias'])

    # In eval mode, changes to importance_threshold will not be propagated to masks.
    operand.importance_threshold = 2
    with torch.no_grad():
        masked_weight, masked_bias = operand(module.weight, module.bias)
        assert torch.allclose(masked_weight, desc['ref_masked_weight'])
        assert torch.allclose(masked_bias, desc['ref_masked_bias'])


@pytest.mark.parametrize(("input_tensor", "threshold", "max_percentile", "ref_output_tensor"), [
    (ensure_tensor([1, 2, 3, 4]), 0.0, 0.9, ensure_tensor([1, 1, 1, 1])),
    (ensure_tensor([1, 2, 3, 4]), 2.5, 0.8, ensure_tensor([0, 0, 1, 1])),
    (ensure_tensor([1, 2, 3, 4]), 2.5, 0.2, ensure_tensor([0, 1, 1, 1])),
    (ensure_tensor([1, 2, 3, 4]), 5.0, 0.8, ensure_tensor([0, 0, 0, 1])),
    (ensure_tensor([1, 1, 1, 1]), 5.0, 0.8, ensure_tensor([0, 0, 0, 0])),
])
@pytest.mark.parametrize('requires_grad', [True, False])
def test_binary_mask_by_threshold(input_tensor, threshold, max_percentile, ref_output_tensor, requires_grad):
    input_tensor.requires_grad_(requires_grad)
    output_tensor = binary_mask_by_threshold(input_tensor, threshold, max_percentile)
    assert torch.allclose(output_tensor, ref_output_tensor)
    assert output_tensor.requires_grad is requires_grad


desc_test_importance_loss = {
    '3layers_with_penalty_lambda': dict(
        sparse_layers_loss=(1., 2., 3.),
        penalty_scheduler_retval=1.5,
        ref_output=3.),
    '3layers_no_penalty_lambda': dict(
        sparse_layers_loss=(1., 2., 3.),
        penalty_scheduler_retval=None,
        ref_output=2.),
    '1layer_with_penalty_lambda': dict(
        sparse_layers_loss=(1.,),
        penalty_scheduler_retval=2.,
        ref_output=2.),
    'no_layer_with_penalty_lambda': dict(
        sparse_layers_loss=(),
        penalty_scheduler_retval=2.,
        ref_output=0.),
    'no_layer_no_penalty_lambda': dict(
        sparse_layers_loss=(),
        penalty_scheduler_retval=None,
        ref_output=0.),
}


class TestImportanceLoss:
    @pytest.mark.parametrize('desc', desc_test_importance_loss.values(), ids=desc_test_importance_loss.keys())
    @pytest.mark.parametrize('requires_grad', [True, False])
    def test_importance_loss_forward(self, desc, requires_grad: bool):
        sparse_layers_loss = desc['sparse_layers_loss']
        penalty_scheduler_retval = desc['penalty_scheduler_retval']
        ref_output = desc['ref_output']
        sparse_layers = [Mock(loss=Mock(return_value=torch.tensor(loss_val, requires_grad=requires_grad)))
                         for loss_val in sparse_layers_loss]
        penalty_scheduler = None
        if penalty_scheduler_retval is not None:
            penalty_scheduler = Mock(current_importance_lambda=penalty_scheduler_retval)
        loss = ImportanceLoss(sparse_layers, penalty_scheduler)
        output = loss()
        for sparse_layer in sparse_layers:
            assert sparse_layer.method_calls == [call.loss()]
        if not sparse_layers_loss:
            assert output == approx(0.)
        else:
            assert isinstance(output, torch.Tensor)
            assert output.requires_grad is requires_grad
            assert torch.allclose(output, torch.tensor(ref_output))


class TestMovementSparsityStatistics:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.importance_threshold = 1.0
        self.importance_regularization_factor = 2.0
        summary = SparsifiedLayerSummary('layer', [1, 1], 0.5, 0.5)
        model_stats = SparsifiedModelStatistics(0.25, 0.5, [summary])
        movement_stats = MovementSparsityStatistics(model_stats, self.importance_threshold,
                                                    self.importance_regularization_factor)
        self.movement_stats = movement_stats

    def test_movement_sparsity_statistics_string(self):
        movement_stats = self.movement_stats
        output_str = movement_stats.to_str()
        assert movement_stats.to_str() in output_str
        assert create_table(
            header=['Statistic\'s name', 'Value'],
            rows=[['Mask Importance Threshold', self.importance_threshold],
                  ['Importance Regularization Factor', self.importance_regularization_factor]]
        ) in output_str

    def test_nncf_stats_can_register_movement_sparsity_stats(self):
        movement_stats = self.movement_stats
        nncf_stats = NNCFStatistics()
        nncf_stats.register('movement_sparsity', movement_stats)
        assert hasattr(nncf_stats, 'movement_sparsity')
        assert nncf_stats.movement_sparsity == movement_stats


class TestSparseConfigByScope:
    @pytest.mark.parametrize('config', [
        {
            "target_scopes": "{re}fine"
        },
        {
            "mode": "fine",
            "target_scopes": "{re}fine"
        },
        {
            "mode": "fine",
            "sparse_factors": [1, 1],
            "target_scopes": "{re}fine"
        },
        {
            "mode": "block",
            "sparse_factors": [32, 32],
            "target_scopes": "{re}block"
        },
        {
            "mode": "per_dim",
            "axis": 0,
            "target_scopes": "{re}prune_row"
        },
        {
            "mode": "per_dim",
            "axis": 1,
            "target_scopes": "prune_column"
        }
    ])
    def test_create_sparse_config_by_scope(self, config: Dict[str, Any]):
        sparse_config_by_scope = SparseConfigByScope.from_config(config)
        assert isinstance(sparse_config_by_scope, SparseConfigByScope)
        ref_target_scopes = config.pop('target_scopes')
        assert sorted(sparse_config_by_scope.target_scopes) == sorted(ref_target_scopes)
        sparse_config = sparse_config_by_scope.sparse_config
        ref_mode = config.pop('mode', 'fine')
        if ref_mode == 'fine':
            ref_sparse_config = dict(mode=SparseStructure.FINE,
                                     sparse_factors=(1, 1),
                                     sparse_axis=None)
        elif ref_mode == 'block':
            ref_sparse_config = dict(mode=SparseStructure.BLOCK,
                                     sparse_factors=tuple(config['sparse_factors']),
                                     sparse_axis=None)
        else:
            ref_sparse_config = dict(mode=SparseStructure.PER_DIM,
                                     sparse_factors=None,
                                     sparse_axis=int(config['axis']))
        assert sparse_config.__dict__ == ref_sparse_config
