from unittest.mock import Mock, call

import pytest
import torch
from nncf.common.sparsity.statistics import (MovementSparsityStatistics,
                                             SparsifiedLayerSummary,
                                             SparsifiedModelStatistics)
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import create_table
from nncf.torch import create_compressed_model
from nncf.torch.sparsity.movement.algo import StructuredMask
from nncf.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.torch.sparsity.movement.loss import ImportanceLoss
from pytest import approx
from tests.torch.sparsity.movement.helpers import (ConfigBuilder,
                                                   bert_tiny_unpretrained)
from tests.torch.test_algo_common import BasicLinearTestModel


@pytest.mark.parametrize(("sparse_structure_by_scopes", "init_weight_importance", "init_bias_importance", "ref_masked_weight", "ref_masked_bias"), [
    (["block", [2, 2], "{re}fc"], [[0, 1], [0, 1]], [1, 0], [[0, 0, 2, 3], [0, 0, 6, 7], [0, 0, 10, 11], [0, 0, 14, 15]], [0, 1, 0, 0]),
    (["per_dim", [0], "{re}fc"], [[0], [1], [0], [1]], [1, 1, 0, 0], [[0] * 4, [4, 5, 6, 7], [0] * 4, [12, 13, 14, 15]], [0, 1, 0, 0]),
    (["per_dim", [1], "{re}fc"], [0, 1, 0, 1], [0], [[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11], [0, 13, 0, 15]], [0] * 4),
    (["fine", [1, 1], "{re}fc"], [[0, 1, 1, 1], [0, 1, 1, 1], [1] * 4, [0] * 4], [0, 0, 0, 1], [[0, 1, 2, 3], [0, 5, 6, 7], [8, 9, 10, 11], [0] * 4], [0, 0, 0, 3])
])
def test_sparsifier_forward(tmp_path, sparse_structure_by_scopes, init_weight_importance, init_bias_importance, ref_masked_weight, ref_masked_bias):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[sparse_structure_by_scopes]).build(
        log_dir=tmp_path, input_info=[{"sample_size": [1, 4]}])
    model = BasicLinearTestModel(size=4)
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    module_info = compression_ctrl.sparsified_module_info[0]
    operand = module_info.operand
    model.fc.weight.data.copy_(torch.arange(16).reshape(4, 4).float())
    model.fc.bias.data.copy_(torch.arange(4).float())
    operand._weight_importance.data.copy_(torch.tensor(init_weight_importance).float())
    operand._bias_importance.data.copy_(torch.tensor(init_bias_importance).float())
    operand.masking_threshold = 0.5
    ori_weight, ori_bias = module_info.module.weight, module_info.module.bias
    masked_weight, masked_bias = operand(ori_weight, ori_bias)  # sparsifier forward function
    assert torch.allclose(masked_weight, torch.tensor(ref_masked_weight).float())
    assert torch.allclose(masked_bias, torch.tensor(ref_masked_bias).float())


def test_structured_mask_setter(tmp_path):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[]).build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_unpretrained(), nncf_config)
    ctx: StructuredMask = sorted(compression_ctrl.structured_ctx_by_group[0],
                                 key=lambda ctx: ctx.target_module_node)[0]  # we pick the self attention linear for key
    grid_size = (2, 4)
    # check independent mask
    ref_mask = ctx.sparse_module_info.operand.get_structured_mask(grid_size)
    assert torch.allclose(ctx.independent_structured_mask, ref_mask)
    ref_mask = ref_mask * 2. + 1.
    ctx.independent_structured_mask = ref_mask
    assert torch.allclose(ctx.independent_structured_mask, ref_mask)
    with pytest.raises(ValueError):
        ctx.independent_structured_mask = ref_mask[:1, :1]
    # check dependent mask
    assert ctx.dependent_structured_mask is None
    ref_mask = torch.ones_like(ref_mask)
    ctx.dependent_structured_mask = ref_mask
    assert torch.allclose(ctx.dependent_structured_mask, ref_mask)
    with pytest.raises(ValueError):
        ctx.dependent_structured_mask = ref_mask[:1, :1]


@pytest.mark.parametrize(("input_tensor", "threshold", "max_percentile", "ref_output_tensor"), [
    (torch.tensor([1., 2, 3, 4]), 0., 0.9, torch.tensor([1., 1, 1, 1])),
    (torch.tensor([1., 2, 3, 4]), 2.5, 0.8, torch.tensor([0., 0, 1, 1])),
    (torch.tensor([1., 2, 3, 4]), 2.5, 0.2, torch.tensor([0., 1, 1, 1])),
    (torch.tensor([1., 2, 3, 4]), 5., 0.8, torch.tensor([0., 0, 0, 1])),
    (torch.tensor([1., 1, 1, 1]), 5., 0.8, torch.tensor([0., 0, 0, 0])),
])
def test_binary_mask_by_threshold(input_tensor, threshold, max_percentile, ref_output_tensor):
    for requires_grad in [True, False]:
        input_tensor.requires_grad_(requires_grad)
        output_tensor = binary_mask_by_threshold(input_tensor, threshold, max_percentile)
        assert torch.allclose(output_tensor, ref_output_tensor)
        assert output_tensor.requires_grad is requires_grad


@pytest.mark.parametrize(('sparse_layers_retvals', "penalty_scheduler_retval", "ref_output"), [
    ((1., 2., 3.), 1.5, 3.),
    ((1., 2., 3.), None, 2.),
    ((1.,), 2., 2.),
    ((), 2., 0.),
    ((), None, 0.),
])
def test_importance_loss(sparse_layers_retvals, penalty_scheduler_retval, ref_output):
    for requires_grad in [True, False]:
        sparse_layers = [Mock(loss=Mock(return_value=torch.tensor(val, requires_grad=requires_grad)))
                         for val in sparse_layers_retvals]
        penalty_scheduler = None
        if penalty_scheduler_retval is not None:
            penalty_scheduler = Mock(current_importance_lambda=penalty_scheduler_retval)
        loss = ImportanceLoss(sparse_layers, penalty_scheduler)
        output = loss()
        if not sparse_layers_retvals:
            assert output == approx(0.)
        else:
            assert isinstance(output, torch.Tensor)
            assert output.requires_grad is requires_grad
            assert torch.allclose(output, torch.tensor(ref_output))
        loss.disable()
        assert loss() == approx(0.)
        for sparse_layer in sparse_layers:
            sparse_layer.method_calls == [call.loss(), call.loss(), call.freeze_importance()]


def test_movement_sparsity_statistics():
    summary = SparsifiedLayerSummary('layer', [1, 1], 0.5, 0.5)
    model_stats = SparsifiedModelStatistics(0.25, 0.5, [summary])
    movement_stats = MovementSparsityStatistics(model_stats, 1.0, 2.0)
    output_str = movement_stats.to_str()
    assert movement_stats.to_str() in output_str
    assert create_table(
        header=['Statistic\'s name', 'Value'],
        rows=[['Mask Importance Threshold', 1.0], ['Importance Regularization Factor', 2.0]]
    ) in output_str


def test_nncf_stats_can_register_movement_sparsity_stats():
    summary = SparsifiedLayerSummary('layer', [1, 1], 0.5, 0.5)
    model_stats = SparsifiedModelStatistics(0.25, 0.5, [summary])
    movement_stats = MovementSparsityStatistics(model_stats, 1.0, 2.0)
    nncf_stats = NNCFStatistics()
    nncf_stats.register('movement_sparsity', movement_stats)
    assert hasattr(nncf_stats, 'movement_sparsity')
    assert nncf_stats.movement_sparsity == movement_stats
