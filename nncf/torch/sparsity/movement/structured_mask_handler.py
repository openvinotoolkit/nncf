import itertools
import logging
from copy import deepcopy
from functools import reduce
from typing import Iterable, List, Tuple, Union, Optional

import numpy as np
import torch
from nncf.experimental.torch.search_building_blocks.search_blocks import \
    BuildingBlockType
from nncf.torch.sparsity.base_algo import SparseModuleInfo
from nncf.torch.sparsity.movement.layers import MovementSparsifier
from nncf.experimental.torch.search_building_blocks.search_blocks import BuildingBlock, get_building_blocks, BuildingBlockType, BlockFilteringStrategy

logger = logging.getLogger('nncf')


class SparsifiedModuleInfoGroup:
    def __init__(self, group_id: int,
                 group_type: BuildingBlockType,
                 sparse_module_info: List[SparseModuleInfo]) -> None:
        self.group_id = group_id
        self.group_type = group_type
        self.sparse_module_info = sparse_module_info


def contains_any(tested_str: str,
                 templates: Union[Iterable[str], str]) -> bool:
    templates = [templates] if isinstance(templates, str) else templates
    return any(str(item) in tested_str for item in templates)


class StructuredMaskContext:
    def __init__(self,
                 sparsifier_operand: MovementSparsifier,
                 module_node_name: str,
                 grid_size: Tuple[int, int],
                 ):
        self.sparsifier_operand = sparsifier_operand
        self.module_node_name = module_node_name
        operand_mask: torch.Tensor = sparsifier_operand.weight_ctx.binary_mask   # type: ignore
        self.operand_mask_shape = operand_mask.shape
        self.grid_size = self._resolve_grid_size(grid_size)
        self.structured_mask_shape = torch.Size(dim // grid for dim, grid in zip(self.operand_mask_shape, self.grid_size))
        self._independent_structured_mask = None
        self._dependent_structured_mask = None

    def __repr__(self) -> str:
        return f"<StructuredMaskContext object for \"{self.module_node_name}\">"

    @property
    def independent_structured_mask(self) -> Optional[torch.Tensor]:
        if self._independent_structured_mask is None:
            logger.warning("Independent structured mask has not been calculated. Return None.")
        return self._independent_structured_mask

    @independent_structured_mask.setter
    @torch.no_grad()
    def independent_structured_mask(self, tensor: torch.Tensor):
        if self._independent_structured_mask is None:
            self._independent_structured_mask = tensor.clone()
        else: 
            if self._independent_structured_mask.shape != tensor.shape:
                raise ValueError("Wrong shape about independent structured mask")
            if self._independent_structured_mask.device != tensor.device:
                logger.info('Changing independent_structured_mask device to %s', tensor.device)
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
        if self._dependent_structured_mask is None:
            self._dependent_structured_mask = tensor.clone()
        else:
            if self._dependent_structured_mask.shape != tensor.shape:
                raise ValueError("Wrong shape about dependent structured mask")
            if self._dependent_structured_mask.device != tensor.device:
                logger.info('Changing dependent_structured_mask device to %s', tensor.device)
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


class StructuredMaskHandler:

    def __init__(self,
                 sparsified_module_info_groups: List[SparsifiedModuleInfoGroup],
                 strategy):
        self.sparsified_module_info_groups = sparsified_module_info_groups
        self.strategy = strategy
        self.strategy_by_group_type = strategy.strategy_by_group_type
        self._structured_mask_ctx_by_group_type = self._create_structured_mask_ctx_by_group_type()

    def _get_group_of_prunable_sparsified_module_info(self) -> List[SparsifiedModuleInfoGroup]:
        module_2_sparse_module_info_map = {sparse_info.module: sparse_info for sparse_info in self.sparsified_module_info}
        building_blocks, _ = get_building_blocks(self.model,
                                                 target_block_types=[BuildingBlockType.MSHA, BuildingBlockType.FF],
                                                 block_filter_strategy=BlockFilteringStrategy.KEEP_SMALL,
                                                 hw_fused_ops=True)
        prunable_sparsified_module_info_groups = []
        for group_id, building_block in enumerate(building_blocks):
            sparsified_module_info = []
            for op_addr in building_block.op_addresses:
                if op_addr.operator_name in NNCF_MODULES_OP_NAMES:
                    module = self.model.get_module_by_scope(op_addr.scope_in_model)
                    module_info = module_2_sparse_module_info_map[module]
                    sparsified_module_info.append(module_info)
            prunable_sparsified_module_info_groups.append(
                SparsifiedModuleInfoGroup(group_id,
                                          building_block.block_type,
                                          sparsified_module_info))
        return prunable_sparsified_module_info_groups

    def _create_structured_mask_ctx_by_group_type(self) -> List[Tuple[BuildingBlockType, List[StructuredMaskContext]]]:
        structured_mask_ctx_by_group_type = []
        for group in self.sparsified_module_info_groups:
            group_type = group.group_type
            ctxes = []
            for module_info in group.sparse_module_info:
                for rule in self.strategy_by_group_type[group_type]:
                    if contains_any(module_info.module_node_name, rule.keywords):
                        ctx = StructuredMaskContext(module_info.operand,
                                                    module_info.module_node_name,
                                                    rule.prune_grid)
                        ctxes.append(ctx)
                        break
                else:
                    raise ValueError("Invalid entry, pls debug")
            structured_mask_ctx_by_group_type.append((group.group_type, ctxes))
        print('*' * 30)
        for group_type, ctxes in structured_mask_ctx_by_group_type:
            print(group_type)
            for ctx in ctxes:
                print(ctx)
        return structured_mask_ctx_by_group_type

    def update_independent_structured_mask(self):
        for _, ctxes in self._structured_mask_ctx_by_group_type:
            for ctx in ctxes:
                ctx.update_independent_structured_mask()

    def resolve_dependent_structured_mask(self):
        for group_type, ctxes in self._structured_mask_ctx_by_group_type:
            if group_type not in self.strategy_by_group_type:
                raise ValueError(f"No strucrtured mask strategy for group_type=\"{group_type}\"")
            rule_list = self.strategy_by_group_type[group_type]
            row_prune_keywords = list(itertools.chain.from_iterable(
                rule.keywords for rule in rule_list if rule.prune_by_row is True))
            col_prune_keywords = list(itertools.chain.from_iterable(
                rule.keywords for rule in rule_list if rule.prune_by_row is False))
            row_prune_ctxes = list(filter(lambda ctx: contains_any(ctx.module_node_name, row_prune_keywords), ctxes))
            col_prune_ctxes = list(filter(lambda ctx: contains_any(ctx.module_node_name, col_prune_keywords), ctxes))
            independent_masks = [ctx.independent_structured_mask for ctx in row_prune_ctxes] + \
                [ctx.independent_structured_mask.t() for ctx in col_prune_ctxes]
            coarse_mask = reduce(torch.logical_or, independent_masks).float()
            with torch.no_grad():
                for ctx in row_prune_ctxes:
                    ctx.dependent_structured_mask = coarse_mask
                for ctx in col_prune_ctxes:
                    ctx.dependent_structured_mask = coarse_mask.t()

    def populate_dependent_structured_mask_to_operand(self):
        for _, ctxes in self._structured_mask_ctx_by_group_type:
            for ctx in ctxes:
                ctx.populate_dependent_structured_mask_to_operand()
