import itertools
import logging
from copy import deepcopy
from functools import reduce
from collections import OrderedDict
from typing import Iterable, List, Tuple, Union, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.torch.sparsity.base_algo import SparseModuleInfo
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import STRUCTURED_MASK_STRATEGY
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import StructuredMaskRule
from nncf.experimental.torch.sparsity.movement.structured_mask_strategy import BaseStructuredMaskStrategy
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlockType
from nncf.common.utils.debug import is_debug
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock, get_building_blocks, BuildingBlockType, BlockFilteringStrategy
from nncf.torch.layers import NNCF_MODULES_OP_NAMES, NNCFLinear
from nncf.torch.nncf_network import NNCFNetwork
from nncf.common.graph.layer_attributes import LinearLayerAttributes
import pandas as pd
from nncf.common.graph.graph import NNCFNodeName

logger = logging.getLogger('nncf')

SUPPORTED_NNCF_MODULES = [NNCFLinear]
EXPECTED_NODE_LAYER_ATTRS = [LinearLayerAttributes]


def contains_any(tested_str: str,
                 templates: Union[Iterable[str], str]) -> bool:
    templates = [templates] if isinstance(templates, str) else templates
    return any(str(item) in tested_str for item in templates)


class StructuredMaskContextStatistics:
    def __init__(self,
                 weight_shape: Tuple[int, int],
                 pruned_weight_shape: Tuple[int, int],
                 bias_shape: Tuple[int],
                 pruned_bias_shape: Tuple[int],
                 head_or_channel_id_to_keep: Union[str, List[int]],
                 module_node_name: NNCFNodeName,
                 ):
        self.weight_shape = weight_shape
        self.pruned_weight_shape = pruned_weight_shape
        self.bias_shape = bias_shape
        self.pruned_bias_shape = pruned_bias_shape
        self.head_or_channel_id_to_keep = head_or_channel_id_to_keep
        self.module_node_name = module_node_name


class StructuredMaskContext:
    def __init__(self,
                 sparsifier_operand: MovementSparsifier,
                 module_node_name: NNCFNodeName,
                 grid_size: Tuple[int, int],
                 prune_by_row: bool,
                 ):
        self.sparsifier_operand = sparsifier_operand
        self.module_node_name = module_node_name
        operand_mask: torch.Tensor = sparsifier_operand.weight_ctx.binary_mask   # type: ignore
        self.operand_mask_shape = operand_mask.shape
        self.grid_size = self._resolve_grid_size(grid_size)
        self.structured_mask_shape = torch.Size(dim // grid for dim, grid in
                                                zip(self.operand_mask_shape, self.grid_size))
        self.prune_by_row = prune_by_row
        self._independent_structured_mask = None
        self._dependent_structured_mask = None

    def __str__(self) -> str:
        prune_info = 'row prune' if self.prune_by_row else 'column prune'
        return f'<{self.__class__.__name__}({prune_info} by {self.grid_size}) for "{self.module_node_name}">'

    @property
    def independent_structured_mask(self) -> Optional[torch.Tensor]:
        if self._independent_structured_mask is None:
            logger.warning("Independent structured mask has not been calculated. Return None.")
        return self._independent_structured_mask

    @independent_structured_mask.setter
    @torch.no_grad()
    def independent_structured_mask(self, tensor: torch.Tensor):
        if self.structured_mask_shape != tensor.shape:
            raise ValueError("Wrong shape about independent structured mask")
        if self._independent_structured_mask is None:
            self._independent_structured_mask = tensor.clone()
        else:
            if self._independent_structured_mask.device != tensor.device:
                logger.warning('Changing independent_structured_mask device to %s', tensor.device)
                self._independent_structured_mask = self._independent_structured_mask.to(tensor.device)
            self._independent_structured_mask.copy_(tensor)

    @property
    def dependent_structured_mask(self) -> torch.Tensor:
        if self._dependent_structured_mask is None:
            logger.warning("Dependent structured mask has not been calculated. Return None.")
        return self._dependent_structured_mask

    @dependent_structured_mask.setter
    @torch.no_grad()
    def dependent_structured_mask(self, tensor: torch.Tensor):
        if self.structured_mask_shape != tensor.shape:
            raise ValueError("Wrong shape about dependent structured mask")
        if self._dependent_structured_mask is None:
            self._dependent_structured_mask = tensor.clone()
        else:
            if self._dependent_structured_mask.device != tensor.device:
                logger.warning('Changing dependent_structured_mask device to %s', tensor.device)
                self._dependent_structured_mask = self._dependent_structured_mask.to(tensor.device)
            self._dependent_structured_mask.copy_(tensor)

    def _resolve_grid_size(self, grid_size) -> Tuple[int, int]:
        a, b = grid_size
        return (a if a > 0 else self.operand_mask_shape[0],
                b if b > 0 else self.operand_mask_shape[1])

    @torch.no_grad()
    def update_independent_structured_mask(self):
        # TODO: Logic here will change later.
        grain_size = self.grid_size
        structured_mask_shape = [dim // grain_size[axes] for axes, dim in enumerate(list(self.sparsifier_operand.weight_ctx.binary_mask.shape))]
        temp_shape = list(itertools.chain(*zip(list(structured_mask_shape), list(grain_size))))
        structured_mask = self.sparsifier_operand.weight_ctx.binary_mask.detach().clone()
        structured_mask = structured_mask.reshape(temp_shape)
        structured_mask = structured_mask.amax(dim=(tuple((np.arange(len(self.sparsifier_operand.weight_ctx.binary_mask.shape)) * 2 + 1))))
        if self.sparsifier_operand.prune_bias is True:
            structured_bias_mask_shape = structured_mask_shape[0]
            structured_bias_mask = self.sparsifier_operand.bias_ctx.binary_mask.detach().clone()
            structured_bias_mask = structured_bias_mask.reshape((structured_bias_mask_shape, -1))
            structured_bias_mask = structured_bias_mask.amax(dim=1)
            # dim_aligned = structured_bias_mask.repeat(structured_mask.shape[1]).reshape(-1, structured_mask.shape[1])
            # structured_mask = structured_mask.logical_or(dim_aligned).to(torch.float32)
            prunable_rows = structured_bias_mask.logical_or(structured_mask.amax(dim=1))  # preserve a row when either bias mask is 1 or weight mask row amax is 1
            prunable_cols = structured_mask.amax(dim=0)
            structured_mask = prunable_rows.unsqueeze(1) * prunable_cols.unsqueeze(0)
        self.independent_structured_mask = structured_mask
        return structured_mask

    def _inflate_structured_mask(self, structured_mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        assert len(structured_mask.shape) == len(grid_size), "Unmatching dimension"
        inflated_mask = structured_mask.clone()
        for axis, repeat_times in enumerate(grid_size):
            inflated_mask = inflated_mask.repeat_interleave(repeat_times, dim=axis)
        return inflated_mask

    def populate_dependent_structured_mask_to_operand(self):
        structured_mask_inflated = self._inflate_structured_mask(self.dependent_structured_mask, self.grid_size)
        self.sparsifier_operand.weight_ctx.binary_mask = structured_mask_inflated
        if self.sparsifier_operand.prune_bias is True:
            self.sparsifier_operand.bias_ctx.binary_mask = structured_mask_inflated.amax(dim=1)

    def gather_statistics_from_operand(self) -> StructuredMaskContextStatistics:
        # original shape
        node = self.sparsifier_operand.target_module_node
        assert isinstance(node.layer_attributes, tuple(EXPECTED_NODE_LAYER_ATTRS))
        weight_shape: Tuple[int, int] = tuple(list(node.layer_attributes.get_weight_shape()))
        bias_shape: Tuple[int] = (node.layer_attributes.get_bias_shape(),) if self.sparsifier_operand.prune_bias else (0,)

        pruned_weight_shape = list(weight_shape)
        head_id_to_keep = []
        if self.prune_by_row:
            pruneable_rows = self.sparsifier_operand.weight_ctx.binary_mask.amax(dim=1)
            pruned_weight_shape[0] = int(pruneable_rows.count_nonzero().item())
            kept_row_blocks = F.max_pool1d(pruneable_rows.unsqueeze(0), kernel_size=self.grid_size[0]).squeeze(0)
            head_id_to_keep = kept_row_blocks.nonzero().view(-1).cpu().numpy().tolist()
        else:
            pruneable_cols = self.sparsifier_operand.weight_ctx.binary_mask.amax(dim=0)
            pruned_weight_shape[1] = int(pruneable_cols.count_nonzero().item())
            kept_col_blocks = F.max_pool1d(pruneable_cols.unsqueeze(0), kernel_size=self.grid_size[1]).squeeze(0)
            head_id_to_keep = kept_col_blocks.nonzero().view(-1).cpu().numpy().tolist()

        pruned_bias_shape = bias_shape
        if self.sparsifier_operand.prune_bias and self.prune_by_row:
            pruned_bias_shape = (int(self.sparsifier_operand.bias_ctx.binary_mask.count_nonzero().item()),)

        return StructuredMaskContextStatistics(
            weight_shape=weight_shape,
            pruned_weight_shape=tuple(pruned_weight_shape),
            bias_shape=bias_shape,
            pruned_bias_shape=pruned_bias_shape,
            head_or_channel_id_to_keep=head_id_to_keep,
            module_node_name=self.module_node_name
        )


class SparsifiedModuleInfoGroup:
    def __init__(self, group_id: int,
                 group_type: BuildingBlockType,
                 sparse_module_info: List[SparseModuleInfo]) -> None:
        self.group_id = group_id
        self.group_type = group_type
        self.sparse_module_info = sparse_module_info


class StructuredMaskContextGroup:
    def __init__(self, group_id: int,
                 group_type: BuildingBlockType,
                 structured_mask_context_list: List[StructuredMaskContext]) -> None:
        self.group_id = group_id
        self.group_type = group_type
        self.structured_mask_context_list = structured_mask_context_list

    def __str__(self) -> str:
        if len(self.structured_mask_context_list) == 0:
            ctx_str = '[]'
        else:
            ctx_str = '\n\t'.join(map(str, self.structured_mask_context_list))
            ctx_str = f'[\n\t{ctx_str}\n]'
        return f'[{self.group_id}]{self.group_type}: {ctx_str}'


class StructuredMaskHandler:

    def __init__(self,
                 compressed_model: NNCFNetwork,
                 sparsified_module_info_list: List[SparseModuleInfo],
                 strategy: BaseStructuredMaskStrategy):
        self.strategy = strategy
        self.strategy_by_group_type = strategy.strategy_by_group_type
        self.compressed_model = compressed_model
        self.sparsified_module_info_list = sparsified_module_info_list

        self._sparsified_module_info_groups = self._get_prunable_sparsified_module_info_group(compressed_model,
                                                                                              sparsified_module_info_list)
        self._structured_mask_ctx_groups = self._create_structured_mask_context_groups(
            self._sparsified_module_info_groups,
            self.strategy_by_group_type)

        logging_str_l = ['Structured mask contexts by group:']
        for group in self._structured_mask_ctx_groups:
            logging_str_l.append(str(group))
        logging.info('\n'.join(logging_str_l))

    @staticmethod
    def _get_prunable_sparsified_module_info_group(
            compressed_model: NNCFNetwork,
            sparsified_module_info_list: List[SparseModuleInfo],
    ) -> List[SparsifiedModuleInfoGroup]:
        module_2_sparse_module_info_map = {minfo.module: minfo for minfo in sparsified_module_info_list}
        building_blocks, _ = get_building_blocks(compressed_model,
                                                 target_block_types=[BuildingBlockType.MSHA, BuildingBlockType.FF],
                                                 block_filter_strategy=BlockFilteringStrategy.KEEP_SMALL,
                                                 hw_fused_ops=True)
        groups = []
        for group_id, building_block in enumerate(building_blocks):
            sparsified_module_info = []
            for op_addr in building_block.op_addresses:
                if op_addr.operator_name in [m.op_func_name for m in SUPPORTED_NNCF_MODULES]:
                    module = compressed_model.get_module_by_scope(op_addr.scope_in_model)
                    module_info = module_2_sparse_module_info_map[module]
                    sparsified_module_info.append(module_info)
            groups.append(SparsifiedModuleInfoGroup(group_id,
                                                    building_block.block_type,
                                                    sparsified_module_info))
        return groups

    @staticmethod
    def _create_structured_mask_context_groups(
        sparsified_module_info_groups: List[SparsifiedModuleInfoGroup],
        rule_by_group_type: Dict[BuildingBlockType, List[StructuredMaskRule]]
    ) -> List[StructuredMaskContextGroup]:
        groups = []
        for group in sparsified_module_info_groups:
            group_type = group.group_type
            group_id = group.group_id
            ctxes = []
            for minfo in group.sparse_module_info:
                for rule in rule_by_group_type[group_type]:
                    if contains_any(minfo.module_node_name, rule.keywords):
                        ctx = StructuredMaskContext(minfo.operand,
                                                    minfo.module_node_name,
                                                    rule.prune_grid,
                                                    rule.prune_by_row)
                        ctxes.append(ctx)
                        break
                else:
                    raise ValueError("No structured mask rule found for "
                                     f"[{group_type}]{minfo.module_node_name}.")
            groups.append(StructuredMaskContextGroup(group_id, group_type, ctxes))
        return groups

    def update_independent_structured_mask(self):
        for group in self._structured_mask_ctx_groups:
            for ctx in group.structured_mask_context_list:
                ctx.update_independent_structured_mask()

    def resolve_dependent_structured_mask(self):
        for group in self._structured_mask_ctx_groups:
            group_type = group.group_type
            if group_type not in self.strategy_by_group_type:
                raise ValueError(f"No strucrtured mask strategy for group_type=\"{group_type}\"")
            ctxes = group.structured_mask_context_list
            row_prune_ctxes = list(filter(lambda ctx: ctx.prune_by_row, ctxes))
            col_prune_ctxes = list(filter(lambda ctx: not ctx.prune_by_row, ctxes))
            independent_masks = [ctx.independent_structured_mask for ctx in row_prune_ctxes] + \
                [ctx.independent_structured_mask.t() for ctx in col_prune_ctxes]
            coarse_mask = reduce(torch.logical_or, independent_masks).float()
            with torch.no_grad():
                for ctx in row_prune_ctxes:
                    ctx.dependent_structured_mask = coarse_mask
                for ctx in col_prune_ctxes:
                    ctx.dependent_structured_mask = coarse_mask.t()

    def populate_dependent_structured_mask_to_operand(self):
        for group in self._structured_mask_ctx_groups:
            for ctx in group.structured_mask_context_list:
                ctx.populate_dependent_structured_mask_to_operand()

    def _gather_statistics_dataframe(self, max_num_of_kept_heads_to_report=20) -> pd.DataFrame:
        module_2_name = {module: name for name, module in self.compressed_model.named_modules()}
        entry_list = []
        for group in self._structured_mask_ctx_groups:
            ctxes = sorted(group.structured_mask_context_list,
                           key=lambda ctx: ctx.sparsifier_operand.target_module_node.node_id)
            for ctx in ctxes:
                stats = ctx.gather_statistics_from_operand()
                if len(stats.head_or_channel_id_to_keep) > max_num_of_kept_heads_to_report:  # avoid too long display
                    stats.head_or_channel_id_to_keep = f'[{len(stats.head_or_channel_id_to_keep)} items]'
                module = self.compressed_model.get_containing_module(stats.module_node_name)
                torch_module_name = module_2_name[module]
                entry_list.append(dict(
                    group_id=group.group_id,
                    type=group.group_type.value,
                    torch_module=torch_module_name,
                    **stats.__dict__
                ))
        return pd.DataFrame(entry_list)

    def report_structured_sparsity(self,
                                   save_dir,
                                   file_name='structured_sparsity',
                                   to_csv: bool = False,
                                   to_markdown: bool = True,
                                   max_num_of_kept_heads_to_report: int = 20) -> pd.DataFrame:
        df = self._gather_statistics_dataframe(max_num_of_kept_heads_to_report)
        if to_csv:
            df.to_csv(Path(save_dir, f'{file_name}.csv'))
        if to_markdown:
            df.to_markdown(Path(save_dir, f'{file_name}.md'))
        return df
