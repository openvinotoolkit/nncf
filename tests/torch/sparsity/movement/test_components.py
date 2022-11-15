from unittest.mock import Mock, call

import pytest
import torch
from nncf.common.sparsity.statistics import (MovementSparsityStatistics,
                                             SparsifiedLayerSummary,
                                             SparsifiedModelStatistics)
from nncf.common.statistics import NNCFStatistics
from nncf.common.utils.helpers import create_table
from nncf.torch import create_compressed_model
from nncf.experimental.torch.sparsity.movement.functions import binary_mask_by_threshold
from nncf.experimental.torch.sparsity.movement.loss import ImportanceLoss
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskContextGroup, StructuredMaskHandler
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import STRUCTURED_MASK_STRATEGY
from pytest import approx
from tests.torch.sparsity.movement.helpers import (ConfigBuilder,
                                                   bert_tiny_unpretrained)
from tests.torch.test_algo_common import BasicLinearTestModel
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
import numpy as np
from collections import OrderedDict


@pytest.mark.parametrize(("sparse_structure_by_scopes", "init_weight_importance", "init_bias_importance", "ref_masked_weight", "ref_masked_bias"), [
    ({"mode": "block", "sparse_factors": [2, 2], "target_scopes": "{re}fc"}, [[0, 1], [0, 1]], [1, 0],
     [[0, 0, 2, 3], [0, 0, 6, 7], [0, 0, 10, 11], [0, 0, 14, 15]], [0, 1, 0, 0]),
    ({"mode": "per_dim", "axis": 0, "target_scopes": "{re}fc"}, [[0], [1], [0], [1]], [1, 1, 0, 0],
     [[0] * 4, [4, 5, 6, 7], [0] * 4, [12, 13, 14, 15]], [0, 1, 0, 0]),
    ({"mode": "per_dim", "axis": 1, "target_scopes": "{re}fc"}, [0, 1, 0, 1], [0],
     [[0, 1, 0, 3], [0, 5, 0, 7], [0, 9, 0, 11], [0, 13, 0, 15]], [0] * 4),
    ({"mode": "fine", "sparse_factors": [1, 1], "target_scopes": "{re}fc"},
     [[0, 1, 1, 1], [0, 1, 1, 1], [1] * 4, [0] * 4], [0, 0, 0, 1],
     [[0, 1, 2, 3], [0, 5, 6, 7], [8, 9, 10, 11], [0] * 4], [0, 0, 0, 3])
])
def test_sparsifier_forward(tmp_path, sparse_structure_by_scopes, init_weight_importance, init_bias_importance, ref_masked_weight, ref_masked_bias):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[sparse_structure_by_scopes], enable_structured_masking=False).build(
        log_dir=tmp_path, input_info=[{"sample_size": [1, 4]}])
    model = BasicLinearTestModel(size=4)
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)
    compressed_model.train()
    minfo = compression_ctrl.sparsified_module_info[0]
    operand = minfo.operand
    model.fc.weight.data.copy_(torch.arange(16).reshape(4, 4).float())
    model.fc.bias.data.copy_(torch.arange(4).float())
    operand.weight_importance.data.copy_(torch.tensor(init_weight_importance).float())
    operand.bias_importance.data.copy_(torch.tensor(init_bias_importance).float())
    operand.importance_threshold = 0.5
    ori_weight, ori_bias = minfo.module.weight, minfo.module.bias
    masked_weight, masked_bias = operand(ori_weight, ori_bias)  # sparsifier forward function
    # TODO: add requires_grad check for operand forward in train/test
    assert torch.allclose(masked_weight, torch.tensor(ref_masked_weight).float())
    assert torch.allclose(masked_bias, torch.tensor(ref_masked_bias).float())


@pytest.mark.skip(reason="temporarily skip due to active refactoring to this")
def test_structured_mask_setter(tmp_path):
    nncf_config = ConfigBuilder(sparse_structure_by_scopes=[]).build(log_dir=tmp_path)
    compression_ctrl, compressed_model = create_compressed_model(bert_tiny_unpretrained(), nncf_config)
    ctx = sorted(compression_ctrl._structured_mask_handler[0][1],
                 key=lambda ctx: ctx.target_module_node)[0]  # we pick the self attention linear for key
    ctx.update_independent_structured_mask()
    grid_size = (2, 4)
    # check independent mask
    ref_mask = ctx.sparsifier_operand.weight_ctx.binary_mask
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


class LayerParam:
    def __init__(self, dtype=torch.float, device=torch.device('cpu'), **kwargs):
        for name, value in kwargs.items():
            setattr(self, name, self.ensure_tensor(value, dtype, device) if value is not None else None)

    @staticmethod
    def ensure_tensor(value, dtype, device):
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=dtype, device=device)
        elif isinstance(value, torch.Tensor):
            return value.to(dtype=dtype, device=device)
        else:
            return torch.tensor(value, dtype=dtype, device=device)


class TransformerLayerMaskParam:
    def __init__(self, MHSA_Q: LayerParam,
                 MHSA_K: LayerParam,
                 MHSA_V: LayerParam,
                 MHSA_O: LayerParam,
                 FFN_I: LayerParam,
                 FFN_O: LayerParam):
        self.MHSA_Q = MHSA_Q
        self.MHSA_K = MHSA_K
        self.MHSA_V = MHSA_V
        self.MHSA_O = MHSA_O
        self.FFN_I = FFN_I
        self.FFN_O = FFN_O

    @property
    def params_in_transformer_block_order(self):
        return [self.MHSA_Q, self.MHSA_K, self.MHSA_V,
                self.MHSA_O, self.FFN_I, self.FFN_O]


structured_mask_desc_prune_1head_1channel = dict(
    unstructured=TransformerLayerMaskParam(
        MHSA_Q=LayerParam(weight=[[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bias=[0, 0, 0, 0]),
        MHSA_K=LayerParam(weight=[[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bias=[1, 0, 0, 0]),
        MHSA_V=LayerParam(weight=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bias=[0, 1, 0, 0]),
        MHSA_O=LayerParam(weight=[[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], bias=[0, 0, 0, 0]),
        FFN_I=LayerParam(weight=[[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]], bias=[1, 0, 0]),
        FFN_O=LayerParam(weight=[[0, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]], bias=[0, 0, 0, 0])
    ),
    independent_structured=TransformerLayerMaskParam(
        MHSA_Q=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        MHSA_K=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        MHSA_V=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        MHSA_O=LayerParam(weight=[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]),
        FFN_I=LayerParam(weight=[[1, 1, 0, 1], [1, 1, 0, 1], [0, 0, 0, 0]]),
        FFN_O=LayerParam(weight=[[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    ),
    dependent_structured=TransformerLayerMaskParam(
        MHSA_Q=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        MHSA_K=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        MHSA_V=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        MHSA_O=LayerParam(weight=[[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]]),
        FFN_I=LayerParam(weight=[[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]),
        FFN_O=LayerParam(weight=[[1, 1, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]])
    ),
)

structured_mask_desc = [structured_mask_desc_prune_1head_1channel]
# run_recipes = [Wav2Vec2RunRecipe(), BertRunRecipe()]
run_recipes = [BertRunRecipe()]


@pytest.mark.parametrize('run_recipe', run_recipes,
                         ids=[f'{r.model_family}_run_recipe' for r in run_recipes])
class TestStructuredMaskHandler:
    @pytest.fixture(autouse=True)
    def setup(self, run_recipe):
        self.model = run_recipe.model
        self.nncf_config = run_recipe.nncf_config
        self.compression_ctrl, self.compressed_model = create_compressed_model(self.model, self.nncf_config, dump_graphs=False)
        strategy = STRUCTURED_MASK_STRATEGY.get(run_recipe.model_family).from_compressed_model(self.compressed_model)
        self.handler = StructuredMaskHandler(self.compression_ctrl.prunable_sparsified_module_info_groups, strategy)
        self.run_recipe = run_recipe
        self.all_ctxes = []
        for group in self.handler._structured_mask_ctx_groups:
            self.all_ctxes.extend(group.structured_mask_context_list)

    def test_create_ctx_groups(self):
        handler = self.handler
        run_recipe = self.run_recipe
        assert len(handler._structured_mask_ctx_groups) == 2
        handler._structured_mask_ctx_groups.sort(key=lambda group: group.group_type.value)
        group_ff = handler._structured_mask_ctx_groups[0]
        assert isinstance(group_ff, StructuredMaskContextGroup)
        assert group_ff.group_type == BuildingBlockType.FF
        assert len(group_ff.structured_mask_context_list) == 2
        group_mhsa = handler._structured_mask_ctx_groups[1]
        assert isinstance(group_mhsa, StructuredMaskContextGroup)
        assert group_mhsa.group_type == BuildingBlockType.MSHA
        assert len(group_mhsa.structured_mask_context_list) == 4

    def test_update_independent_structured_mask(self, mocker):
        handler = self.handler
        mock_methods = [mocker.patch.object(ctx, 'update_independent_structured_mask') for ctx in self.all_ctxes]
        handler.update_independent_structured_mask()
        for mock_method in mock_methods:
            mock_method.assert_called_once()

    @pytest.mark.parametrize('desc', structured_mask_desc)
    def test_resolve_dependent_structured_mask(self, desc):
        handler = self.handler
        run_recipe = self.run_recipe
        modules = run_recipe.get_nncf_modules_in_transformer_block_order(self.compressed_model)[0]
        module_2_node_name = {minfo.module: minfo.module_node_name for minfo in self.compression_ctrl.sparsified_module_info}
        node_name_2_context = {ctx.module_node_name: ctx for ctx in self.all_ctxes}
        ctxes = [node_name_2_context[module_2_node_name[m]] for m in modules]
        for ctx, param in zip(ctxes, desc['independent_structured'].params_in_transformer_block_order):
            ctx.independent_structured_mask = param.weight

        handler.resolve_dependent_structured_mask()
        for ctx, ref_param in zip(ctxes, desc['dependent_structured'].params_in_transformer_block_order):
            assert torch.allclose(ctx.dependent_structured_mask, ref_param.weight)

    def test_populate_dependent_structured_mask_to_operand(self, mocker):
        handler = self.handler
        mock_methods = [mocker.patch.object(ctx, 'populate_dependent_structured_mask_to_operand') for ctx in self.all_ctxes]
        handler.populate_dependent_structured_mask_to_operand()
        for mock_method in mock_methods:
            mock_method.assert_called_once()
