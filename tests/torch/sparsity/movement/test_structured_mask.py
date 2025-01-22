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
import re
from pathlib import Path
from typing import List, Tuple
from unittest.mock import Mock

import pandas as pd
import pytest
import torch
from packaging import version

from nncf.common.logging import nncf_logger
from nncf.config import NNCFConfig
from nncf.experimental.torch.sparsity.movement.algo import is_supported_model_family
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.experimental.torch.sparsity.movement.layers import SparseConfig
from nncf.experimental.torch.sparsity.movement.layers import SparseStructure
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskContext
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskContextGroup
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskContextStatistics
from nncf.experimental.torch.sparsity.movement.structured_mask_handler import StructuredMaskHandler
from nncf.torch import create_compressed_model
from tests.torch.sparsity.movement.helpers import BaseMockRunRecipe
from tests.torch.sparsity.movement.helpers import BertRunRecipe
from tests.torch.sparsity.movement.helpers import DictInTransformerBlockOrder
from tests.torch.sparsity.movement.helpers import SwinRunRecipe
from tests.torch.sparsity.movement.helpers import Wav2Vec2RunRecipe
from tests.torch.sparsity.movement.helpers import mock_linear_nncf_node

STRUCTURED_MASK_SUPPORTED_RECIPES = [
    BertRunRecipe().model_config_(hidden_size=4, intermediate_size=3),
    BertRunRecipe().model_config_(hidden_size=4, intermediate_size=3, mhsa_qkv_bias=False),
    BertRunRecipe().model_config_(hidden_size=4, intermediate_size=3, mhsa_o_bias=False),
    BertRunRecipe().model_config_(hidden_size=4, intermediate_size=3, ffn_bias=False),
    BertRunRecipe().model_config_(
        hidden_size=4, intermediate_size=3, mhsa_qkv_bias=False, mhsa_o_bias=False, ffn_bias=False
    ),
    Wav2Vec2RunRecipe().model_config_(hidden_size=4, intermediate_size=3),
    SwinRunRecipe().model_config_(embed_dim=4, mlp_ratio=0.75, qkv_bias=False),
    SwinRunRecipe().model_config_(embed_dim=4, mlp_ratio=0.75, depths=[1], num_heads=[2]),
]

desc_test_update_independent_structured_mask = {
    "prune1row": dict(
        weight_binary_mask=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=torch.FloatTensor([1, 0, 0]),
        prune_grid=(1, 3),
        ref_independent_structured_mask=torch.FloatTensor([[1], [1], [0]]),
    ),
    "prune1col": dict(
        weight_binary_mask=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=torch.FloatTensor([1, 0, 0]),
        prune_grid=(3, 1),
        ref_independent_structured_mask=torch.FloatTensor([[1, 1, 0]]),
    ),
    "prune1col_nobias": dict(
        weight_binary_mask=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=None,
        prune_grid=(3, 1),
        ref_independent_structured_mask=torch.FloatTensor([[1, 1, 0]]),
    ),
    "not_prunable": dict(
        weight_binary_mask=torch.FloatTensor([[1, 1, 0], [1, 1, 0], [0, 0, 0]]),
        bias_binary_mask=torch.FloatTensor([1, 1, 1]),
        prune_grid=(1, 3),
        ref_independent_structured_mask=torch.FloatTensor([[1], [1], [1]]),
    ),
}

desc_test_gather_statistics_from_operand = {
    "row_prune_with_bias": dict(
        weight_mask=torch.FloatTensor([[0] * 4, [0] * 4, [1] * 4, [1] * 4]),
        bias_mask=torch.FloatTensor([0, 0, 1, 1]),
        prune_grid=(2, 4),
        prune_by_row=True,
        pruned_weight_shape=(2, 4),
        pruned_bias_shape=(2,),
        head_to_keep=[1],
    ),
    "row_prune_without_bias": dict(
        weight_mask=torch.FloatTensor([[0] * 4, [0] * 4, [1] * 4, [1] * 4]),
        bias_mask=None,
        prune_grid=(2, 4),
        prune_by_row=True,
        pruned_weight_shape=(2, 4),
        pruned_bias_shape=(0,),
        head_to_keep=[1],
    ),
    "col_prune_with_bias": dict(
        weight_mask=torch.FloatTensor([[1, 1, 1, 0]] * 4),
        bias_mask=torch.FloatTensor([1, 1, 1, 1]),
        prune_grid=(4, 1),
        prune_by_row=False,
        pruned_weight_shape=(4, 3),
        pruned_bias_shape=(4,),
        head_to_keep=[0, 1, 2],
    ),
    "col_prune_without_bias": dict(
        weight_mask=torch.FloatTensor([[1, 1, 1, 0]] * 4),
        bias_mask=None,
        prune_grid=(4, 1),
        prune_by_row=False,
        pruned_weight_shape=(4, 3),
        pruned_bias_shape=(0,),
        head_to_keep=[0, 1, 2],
    ),
}


class TestStructuredMaskContext:
    @pytest.mark.parametrize(
        ("input_grid", "ref_resolved_grid"), [((1, 4), (1, 4)), ((-1, 2), (4, 2)), ((4, -1), (4, 4))]
    )
    def test_can_resolve_prune_grid_size(self, input_grid, ref_resolved_grid):
        operand = MovementSparsifier(
            mock_linear_nncf_node(4, 4),
            SparseConfig(SparseStructure.FINE),
        )
        prune_by_row = input_grid[0] not in [-1, 4]
        ctx = StructuredMaskContext(operand, "linear", input_grid, prune_by_row)
        assert ctx.grid_size == ref_resolved_grid

    @pytest.mark.parametrize(
        ("structure_grid_size", "ref_mask_shape"),
        [
            ((2, 4), torch.Size([2, 1])),
            ((4, 1), torch.Size([1, 4])),
        ],
    )
    @pytest.mark.parametrize("is_dependent_mask", [True, False], ids=["dependent", "independent"])
    def test_structured_mask_setter(self, is_dependent_mask: bool, structure_grid_size, ref_mask_shape):
        mask_name = "dependent_structured_mask" if is_dependent_mask else "independent_structured_mask"
        operand = MovementSparsifier(mock_linear_nncf_node(4, 4))
        prune_by_row = structure_grid_size[0] not in [-1, 4]
        ctx = StructuredMaskContext(operand, "linear", structure_grid_size, prune_by_row)
        assert getattr(ctx, mask_name) is None
        # initialize
        ref_mask1 = torch.ones(ref_mask_shape)
        setattr(ctx, mask_name, ref_mask1)
        assert torch.equal(getattr(ctx, mask_name), ref_mask1)
        assert getattr(ctx, mask_name).requires_grad is False
        assert getattr(ctx, mask_name).device == ref_mask1.device
        id_on_creation = id(getattr(ctx, mask_name))
        assert id_on_creation != id(ref_mask1)
        # reset value
        ref_mask2 = torch.zeros(ref_mask_shape, requires_grad=True)
        setattr(ctx, mask_name, ref_mask2)
        assert torch.equal(getattr(ctx, mask_name), ref_mask2)
        assert getattr(ctx, mask_name).requires_grad is False
        assert id(getattr(ctx, mask_name)) == id_on_creation

    @pytest.mark.parametrize("is_dependent_mask", [True, False], ids=["dependent", "independent"])
    def test_structured_mask_setter_with_wrong_shape(self, is_dependent_mask: bool):
        mask_name = "dependent_structured_mask" if is_dependent_mask else "independent_structured_mask"
        operand = MovementSparsifier(mock_linear_nncf_node(1, 1))
        ctx = StructuredMaskContext(operand, "linear", (1, 1), True)
        with pytest.raises(ValueError, match="Wrong shape"):
            setattr(ctx, mask_name, torch.ones(2))

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("1.12"),
        reason=f"torch {torch.__version__} may not compatible with installed transformers package. "
        f"Some tests may fail with error",
    )
    @pytest.mark.parametrize("is_dependent_mask", [True, False], ids=["dependent", "independent"])
    def test_structured_mask_setter_with_device_change(self, is_dependent_mask: bool, nncf_caplog):
        mask_name = "dependent_structured_mask" if is_dependent_mask else "independent_structured_mask"
        operand = MovementSparsifier(mock_linear_nncf_node(1, 1))
        ctx = StructuredMaskContext(operand, "linear", (1, 1), True)
        setattr(ctx, mask_name, torch.ones(1, 1))
        # use 'meta' device for check since it does not need gpus
        mock_meta_mask = torch.ones((1, 1), device=torch.device("meta"))
        with nncf_caplog.at_level(logging.DEBUG, logger=nncf_logger.name):
            setattr(ctx, mask_name, mock_meta_mask)
            assert getattr(ctx, mask_name).device == torch.device("meta")
        assert f"Changing {mask_name} device" in nncf_caplog.text

    @pytest.mark.parametrize(
        "desc",
        desc_test_update_independent_structured_mask.values(),
        ids=desc_test_update_independent_structured_mask.keys(),
    )
    def test_update_independent_structured_mask(self, desc):
        sparsifier = Mock()
        sparsifier.prune_bias = desc["bias_binary_mask"] is not None
        sparsifier.weight_ctx.binary_mask = desc["weight_binary_mask"]
        if sparsifier.prune_bias:
            sparsifier.bias_ctx.binary_mask = desc["bias_binary_mask"]
        ctx = StructuredMaskContext(sparsifier, "linear", desc["prune_grid"], True)
        ctx.update_independent_structured_mask_from_operand()
        assert torch.equal(ctx.independent_structured_mask, desc["ref_independent_structured_mask"])

    @pytest.mark.parametrize(
        "desc",
        [
            dict(
                mask=torch.FloatTensor([[1, 0, 1]]),
                prune_grid=(2, 1),
                ref_binary_mask=torch.FloatTensor([[1, 0, 1], [1, 0, 1]]),
            ),
            dict(
                mask=torch.FloatTensor([[1], [0], [1]]),
                prune_grid=(1, 2),
                ref_binary_mask=torch.FloatTensor([[1, 1], [0, 0], [1, 1]]),
            ),
        ],
    )
    def test_populate_dependent_structured_mask(self, desc: dict):
        sparsifier = Mock()
        sparsifier.prune_bias = True
        ref_weight_mask = desc["ref_binary_mask"]
        ref_bias_mask = ref_weight_mask.amax(dim=1)
        sparsifier.weight_ctx.binary_mask = torch.ones_like(ref_weight_mask)
        sparsifier.bias_ctx.binary_mask = torch.ones_like(ref_bias_mask)
        ctx = StructuredMaskContext(sparsifier, "linear", desc["prune_grid"], True)
        ctx.dependent_structured_mask = desc["mask"]
        ctx.populate_dependent_structured_mask_to_operand()
        assert torch.equal(sparsifier.weight_ctx.binary_mask, ref_weight_mask)
        assert torch.equal(sparsifier.bias_ctx.binary_mask, ref_bias_mask)

    def test_populate_dependent_structured_mask_by_row_and_column(self):
        mask_1 = torch.FloatTensor([[1, 0, 1]])
        prune_grid_1 = (2, 1)
        mask_2 = torch.FloatTensor([[0], [1]])
        prune_grid_2 = (1, 3)
        ref_weight_mask = torch.FloatTensor([[0, 0, 0], [1, 0, 1]])
        ref_bias_mask = ref_weight_mask.amax(dim=1)

        sparsifier = Mock()
        sparsifier.prune_bias = True

        ref_bias_mask = ref_weight_mask.amax(dim=1)
        sparsifier.weight_ctx.binary_mask = torch.ones_like(ref_weight_mask)
        sparsifier.bias_ctx.binary_mask = torch.ones_like(ref_bias_mask)

        ctx = StructuredMaskContext(sparsifier, "linear", prune_grid_1, True)
        ctx.dependent_structured_mask = mask_1
        ctx.populate_dependent_structured_mask_to_operand()

        ctx = StructuredMaskContext(sparsifier, "linear", prune_grid_2, True)
        ctx.dependent_structured_mask = mask_2
        ctx.populate_dependent_structured_mask_to_operand()

        assert torch.equal(sparsifier.weight_ctx.binary_mask, ref_weight_mask)
        assert torch.equal(sparsifier.bias_ctx.binary_mask, ref_bias_mask)

    @pytest.mark.parametrize(
        "desc", desc_test_gather_statistics_from_operand.values(), ids=desc_test_gather_statistics_from_operand.keys()
    )
    def test_gather_statistics_from_operand(self, desc: dict):
        prune_bias = desc["bias_mask"] is not None
        weight_shape = tuple(desc["weight_mask"].shape)
        bias_shape = tuple(desc["bias_mask"].shape) if prune_bias else (0,)
        node = mock_linear_nncf_node(weight_shape[1], weight_shape[0], bias=prune_bias)
        sparsifier = MovementSparsifier(node, SparseConfig(SparseStructure.FINE))
        sparsifier.weight_ctx.binary_mask = desc["weight_mask"]
        if prune_bias:
            sparsifier.bias_ctx.binary_mask = desc["bias_mask"]
        ctx = StructuredMaskContext(sparsifier, node.node_name, desc["prune_grid"], desc["prune_by_row"])
        ref_stats = StructuredMaskContextStatistics(
            weight_shape=weight_shape,
            pruned_weight_shape=desc["pruned_weight_shape"],
            bias_shape=bias_shape,
            pruned_bias_shape=desc["pruned_bias_shape"],
            head_or_channel_id_to_keep=desc["head_to_keep"],
            module_node_name=node.node_name,
        )
        stats = ctx.gather_statistics_from_operand()
        assert stats.__dict__ == ref_stats.__dict__

    @pytest.mark.parametrize("prune_by_row", [True, False])
    def test_string_representation(self, prune_by_row: bool):
        node = mock_linear_nncf_node(1, 1, node_name="mock_linear")
        grid_size = (1, 1)
        operand = MovementSparsifier(node, SparseConfig(SparseStructure.FINE))
        ctx = StructuredMaskContext(operand, node.node_name, grid_size, prune_by_row)
        row_or_col = "row" if prune_by_row else "column"
        ref_str = f'StructuredMaskContext({row_or_col} prune by {grid_size}, "{node.node_name}")'
        assert str(ctx) == ref_str


class TestStructuredMaskContextGroup:
    @pytest.mark.parametrize("num_contexts", [0, 1, 2])
    def test_string_representation(self, num_contexts: int):
        ctxes = [Mock(__str__=Mock(return_value=f"ctx{i}")) for i in range(num_contexts)]
        ctx_group = StructuredMaskContextGroup(0, ctxes)
        prefix = "StructuredMaskContextGroup[0]: "
        if num_contexts == 0:
            assert str(ctx_group) == f"{prefix}[]"
        else:
            assert str(ctx_group) == "{prefix}[{ctxes}\n]".format(
                prefix=prefix, ctxes="".join(f"\n\tctx{i}" for i in range(num_contexts))
            )


desc_test_resolve_dependent_structured = {
    "prune_1head_1channel": dict(
        independent_structured=DictInTransformerBlockOrder(
            mhsa_q=torch.FloatTensor([[1], [0]]),
            mhsa_k=torch.FloatTensor([[1], [0]]),
            mhsa_v=torch.FloatTensor([[1], [0]]),
            mhsa_o=torch.FloatTensor([[1, 0]]),
            ffn_i=torch.FloatTensor([[1], [1], [0]]),
            ffn_o=torch.FloatTensor([[1, 1, 0]]),
        ),
        dependent_structured=DictInTransformerBlockOrder(
            mhsa_q=torch.FloatTensor([[1], [0]]),
            mhsa_k=torch.FloatTensor([[1], [0]]),
            mhsa_v=torch.FloatTensor([[1], [0]]),
            mhsa_o=torch.FloatTensor([[1, 0]]),
            ffn_i=torch.FloatTensor([[1], [1], [0]]),
            ffn_o=torch.FloatTensor([[1, 1, 0]]),
        ),
    ),
    "prune_0head_0channel": dict(
        independent_structured=DictInTransformerBlockOrder(
            mhsa_q=torch.FloatTensor([[1], [0]]),
            mhsa_k=torch.FloatTensor([[1], [0]]),
            mhsa_v=torch.FloatTensor([[0], [1]]),
            mhsa_o=torch.FloatTensor([[1, 0]]),
            ffn_i=torch.FloatTensor([[1], [1], [0]]),
            ffn_o=torch.FloatTensor([[1, 0, 1]]),
        ),
        dependent_structured=DictInTransformerBlockOrder(
            mhsa_q=torch.FloatTensor([[1], [1]]),
            mhsa_k=torch.FloatTensor([[1], [1]]),
            mhsa_v=torch.FloatTensor([[1], [1]]),
            mhsa_o=torch.FloatTensor([[1, 1]]),
            ffn_i=torch.FloatTensor([[1], [1], [1]]),
            ffn_o=torch.FloatTensor([[1, 1, 1]]),
        ),
    ),
}


class TestStructuredMaskHandler:
    @pytest.mark.parametrize(
        "run_recipe", STRUCTURED_MASK_SUPPORTED_RECIPES, ids=[r.model_family for r in STRUCTURED_MASK_SUPPORTED_RECIPES]
    )
    def test_create_ctx_groups(self, run_recipe):
        compression_ctrl, _ = create_compressed_model(run_recipe.model(), run_recipe.nncf_config(), dump_graphs=False)
        handler, _ = self._get_handler_from_ctrl(compression_ctrl)
        num_transformer_blocks = sum(tbinfo.num_hidden_layers for tbinfo in run_recipe.transformer_block_info)
        assert len(handler._structured_mask_ctx_groups) == num_transformer_blocks * 2
        for i in range(num_transformer_blocks * 2):
            group = handler._structured_mask_ctx_groups[i]
            assert isinstance(group, StructuredMaskContextGroup)
            assert len(group.structured_mask_contexts) in (2, 4)

    def test_update_independent_structured_mask(self, mocker):
        run_recipe = STRUCTURED_MASK_SUPPORTED_RECIPES[0]
        compression_ctrl, _ = create_compressed_model(run_recipe.model(), run_recipe.nncf_config(), dump_graphs=False)
        handler, all_ctxes = self._get_handler_from_ctrl(compression_ctrl)
        mock_methods = [
            mocker.patch.object(ctx, "update_independent_structured_mask_from_operand") for ctx in all_ctxes
        ]
        handler.update_independent_structured_mask()
        for mock_method in mock_methods:
            mock_method.assert_called_once()

    @pytest.mark.parametrize(
        "desc", desc_test_resolve_dependent_structured.values(), ids=desc_test_resolve_dependent_structured.keys()
    )
    def test_resolve_dependent_structured_mask(self, desc):
        run_recipe = STRUCTURED_MASK_SUPPORTED_RECIPES[0]
        compression_ctrl, compressed_model = create_compressed_model(
            run_recipe.model(), run_recipe.nncf_config(), dump_graphs=False
        )
        handler, all_ctxes = self._get_handler_from_ctrl(compression_ctrl)
        module_dict = run_recipe.get_nncf_modules_in_transformer_block_order(compressed_model)[0]
        module_vs_node_name_map = {
            minfo.module: minfo.module_node_name for minfo in compression_ctrl.sparsified_module_info
        }
        node_name_vs_context_map = {ctx.module_node_name: ctx for ctx in all_ctxes}
        ctxes = [node_name_vs_context_map[module_vs_node_name_map[m]] for m in module_dict.values()]
        for ctx, param in zip(ctxes, desc["independent_structured"].values()):
            ctx.independent_structured_mask = param

        handler.resolve_dependent_structured_mask()
        for ctx, ref_param in zip(ctxes, desc["dependent_structured"].values()):
            assert torch.allclose(ctx.dependent_structured_mask, ref_param)

    def test_populate_dependent_structured_mask_to_operand(self, mocker):
        run_recipe = STRUCTURED_MASK_SUPPORTED_RECIPES[0]
        compression_ctrl, _ = create_compressed_model(run_recipe.model(), run_recipe.nncf_config(), dump_graphs=False)
        handler, all_ctxes = self._get_handler_from_ctrl(compression_ctrl)
        mock_methods = [mocker.patch.object(ctx, "populate_dependent_structured_mask_to_operand") for ctx in all_ctxes]
        handler.populate_dependent_structured_mask_to_operand()
        for mock_method in mock_methods:
            mock_method.assert_called_once()

    @pytest.mark.parametrize("max_num_of_kept_heads_to_report", [1, 20])
    def test_report_structured_sparsity(self, tmp_path, mocker, max_num_of_kept_heads_to_report):
        file_name = "structured_report"
        run_recipe = STRUCTURED_MASK_SUPPORTED_RECIPES[0]
        compression_ctrl, _ = create_compressed_model(run_recipe.model(), run_recipe.nncf_config(), dump_graphs=False)
        handler, _ = self._get_handler_from_ctrl(compression_ctrl)
        df = handler.report_structured_sparsity(
            tmp_path, file_name=file_name, to_csv=True, max_num_of_kept_heads_to_report=max_num_of_kept_heads_to_report
        )
        assert isinstance(df, pd.DataFrame)
        columns = df.columns.to_list()
        mock_stat = StructuredMaskContextStatistics(*([mocker.Mock()] * 6))
        ref_columns = ["group_id", "torch_module", *mock_stat.__dict__.keys()]
        assert sorted(columns) == sorted(ref_columns)
        assert len(df) == 6 * sum(tbinfo.num_hidden_layers for tbinfo in run_recipe.transformer_block_info)
        for item in df["head_or_channel_id_to_keep"]:
            if isinstance(item, list):
                assert len(item) <= max_num_of_kept_heads_to_report
            else:
                assert isinstance(item, str)
                assert re.fullmatch(r"\[[0-9]+ items\]", item) is not None
        assert Path(tmp_path, f"{file_name}.csv").is_file()

    def _get_handler_from_ctrl(self, compression_ctrl) -> Tuple[StructuredMaskHandler, List[StructuredMaskContext]]:
        handler = compression_ctrl._structured_mask_handler
        all_ctxes = []
        for group in handler._structured_mask_ctx_groups:
            all_ctxes.extend(group.structured_mask_contexts)
        return handler, all_ctxes


class TestStructuredMaskStrategy:
    @pytest.mark.parametrize("run_recipe", STRUCTURED_MASK_SUPPORTED_RECIPES)
    def test_detect_supported_model_family(self, run_recipe: BaseMockRunRecipe):
        empty_nncf_config = NNCFConfig(input_info=run_recipe.dumps_model_input_info())
        _, compressed_model = create_compressed_model(run_recipe.model(), empty_nncf_config, dump_graphs=False)
        is_supported = is_supported_model_family(compressed_model)
        assert is_supported == run_recipe.supports_structured_masking
