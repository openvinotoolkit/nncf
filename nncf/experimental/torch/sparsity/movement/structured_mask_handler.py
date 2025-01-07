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
from functools import reduce
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
import torch.nn.functional as F

from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.logging import nncf_logger
from nncf.experimental.common.pruning.nodes_grouping import get_pruning_groups
from nncf.experimental.common.pruning.nodes_grouping import select_largest_groups
from nncf.experimental.common.pruning.propagation_data import ProducerInfo
from nncf.experimental.torch.pruning.operations import PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES
from nncf.experimental.torch.sparsity.movement.layers import MovementSparsifier
from nncf.experimental.torch.sparsity.movement.layers import SparseStructure
from nncf.torch.layers import NNCFLinear
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.sparsity.base_algo import SparseModuleInfo

SUPPORTED_NNCF_MODULES = [NNCFLinear]
EXPECTED_NODE_LAYER_ATTRS = [LinearLayerAttributes]


class StructuredMaskContextStatistics:
    """
    Describes details of the resolved structured mask in a supported layer.
    """

    def __init__(
        self,
        weight_shape: Tuple[int, int],
        pruned_weight_shape: Tuple[int, int],
        bias_shape: Tuple[int],
        pruned_bias_shape: Tuple[int],
        head_or_channel_id_to_keep: List[int],
        module_node_name: NNCFNodeName,
    ):
        """
        Initializes the statistics for the target linear module of a structured mask context.

        :param weight_shape: Shape of the original weight in a linear layer.
        :param pruned_weight_shape: Shape of the weight after structured mask resolution,
            discarding the pruned regions.
        :param bias_shape: Shape of the original bias in a linear layer.
        :param pruned_bias_shape: Shape of the bias after structured mask resolution,
            discarding the pruned regions.
        :param head_or_channel_id_to_keep: If the layer belongs to multi-head self-attention block,
            specifies the preserved head ids after structured masking. If the layer belongs to feed-
            forward network, specifies the preserved channels after structured masking.
        :param module_node_name: Node name of the target module.
        """
        self.weight_shape = weight_shape
        self.pruned_weight_shape = pruned_weight_shape
        self.bias_shape = bias_shape
        self.pruned_bias_shape = pruned_bias_shape
        self.head_or_channel_id_to_keep = head_or_channel_id_to_keep
        self.module_node_name = module_node_name


class StructuredMaskContext:
    """
    Context to interact with the operand of a module in movement sparsity.

    This context can resolve the independent structured mask from operand, and can refresh the binary
    mask back to operand with dependent structured mask. Serves as an agent for `StructuredMaskHandler`
    to conduct structured mask resolution.
    """

    def __init__(
        self,
        sparsifier_operand: MovementSparsifier,
        module_node_name: NNCFNodeName,
        grid_size: Tuple[int, int],
        prune_by_row: bool,
    ):
        """
        Initializes the context of the target module for structured masking.

        :param sparsifier_operand: Operand for the target module.
        :param module_node_name: Node name of the target module.
        :param grid_size: The grid shape for resolving the independent structured mask.
        :param prune_by_row: Determines whether to resolve the independent structured mask by row or column.
        """
        self.sparsifier_operand = sparsifier_operand
        self.module_node_name = module_node_name
        operand_mask: torch.Tensor = sparsifier_operand.weight_ctx.binary_mask
        self.operand_mask_shape = operand_mask.shape
        self.grid_size = self._resolve_grid_size(grid_size)
        self.structured_mask_shape = torch.Size(
            dim // grid for dim, grid in zip(self.operand_mask_shape, self.grid_size)
        )
        self.prune_by_row = prune_by_row
        self._independent_structured_mask = None
        self._dependent_structured_mask = None

    def __str__(self) -> str:
        prune_info = "row prune" if self.prune_by_row else "column prune"
        return f'{self.__class__.__name__}({prune_info} by {self.grid_size}, "{self.module_node_name}")'

    @property
    def independent_structured_mask(self) -> Optional[torch.Tensor]:
        if self._independent_structured_mask is None:
            nncf_logger.debug("Independent structured mask has not been calculated. Return None.")
        return self._independent_structured_mask

    @independent_structured_mask.setter
    @torch.no_grad()
    def independent_structured_mask(self, tensor: torch.Tensor):
        if self.structured_mask_shape != tensor.shape:
            raise ValueError("Wrong shape about independent structured mask.")
        if self._independent_structured_mask is None:
            self._independent_structured_mask = tensor.clone()
        else:
            if self._independent_structured_mask.device != tensor.device:
                nncf_logger.debug(f"Changing independent_structured_mask device to {tensor.device}")
                self._independent_structured_mask = self._independent_structured_mask.to(tensor.device)
            self._independent_structured_mask.copy_(tensor)

    @property
    def dependent_structured_mask(self) -> Optional[torch.Tensor]:
        if self._dependent_structured_mask is None:
            nncf_logger.debug("Dependent structured mask has not been calculated. Return None.")
        return self._dependent_structured_mask

    @dependent_structured_mask.setter
    @torch.no_grad()
    def dependent_structured_mask(self, tensor: torch.Tensor):
        if self.structured_mask_shape != tensor.shape:
            raise ValueError("Wrong shape about dependent structured mask.")
        if self._dependent_structured_mask is None:
            self._dependent_structured_mask = tensor.clone()
        else:
            if self._dependent_structured_mask.device != tensor.device:
                nncf_logger.debug(
                    f"Changing dependent_structured_mask device to {tensor.device}",
                )
                self._dependent_structured_mask = self._dependent_structured_mask.to(tensor.device)
            self._dependent_structured_mask.copy_(tensor)

    @torch.no_grad()
    def update_independent_structured_mask_from_operand(self):
        """
        Gets the current unstructured binary mask from operand, resolves it to the independent structured one, and
        stores in `self.independent_structured_mask` for later use in `StructuredMaskHandler`.
        """
        weight_binary_mask = self.sparsifier_operand.weight_ctx.binary_mask.detach().clone()
        mask_by_grid = F.max_pool2d(
            weight_binary_mask.unsqueeze(0), kernel_size=self.grid_size, stride=self.grid_size
        ).squeeze(0)
        preserved_cols = mask_by_grid.amax(dim=0)
        preserved_rows = mask_by_grid.amax(dim=1)

        if self.sparsifier_operand.prune_bias:
            bias_binary_mask = self.sparsifier_operand.bias_ctx.binary_mask.detach().clone()
            bias_preserved_rows = F.max_pool1d(
                bias_binary_mask.view(1, -1), kernel_size=self.grid_size[0], stride=self.grid_size[0]
            ).squeeze(0)
            preserved_rows = bias_preserved_rows.logical_or(preserved_rows)

        structured_mask = preserved_rows.unsqueeze(1) * preserved_cols.unsqueeze(0)
        self.independent_structured_mask = structured_mask
        return structured_mask

    def initialize_binary_mask(self):
        """
        Initialize binary mask by all ones. The inflated dependent mask will be applied via logical "and"
        operation to it. It's needed for the case when 1 binary mask shared for 2 groups: in one group operator
        can be pruned by input channels, i.e. be a consumer of pruning masks, and for another - can be pruned by
        output channels, i.e. be a producer of pruning masks.
        Initial  |  Mask 2 last input channels   |  Mask middle output channel    |     Result
        --------------------------------------------------------------------------------------
         1111                 1100                             1111                     1100
         1111    &            1100               &             0000                =    0000
         1111                 1100                             1111                     1100
        """

        self.sparsifier_operand.weight_ctx.binary_mask.fill_(1)
        if self.sparsifier_operand.prune_bias:
            self.sparsifier_operand.bias_ctx.binary_mask.fill_(1)

    def populate_dependent_structured_mask_to_operand(self):
        """
        Updates the actual binary masks in operand with `self.dependent_structured_mask`.
        """
        structured_mask_inflated = self._inflate_structured_mask(self.dependent_structured_mask, self.grid_size)
        self.sparsifier_operand.weight_ctx.binary_mask *= structured_mask_inflated
        if self.sparsifier_operand.prune_bias:
            self.sparsifier_operand.bias_ctx.binary_mask *= structured_mask_inflated.amax(dim=1)

    def gather_statistics_from_operand(self) -> StructuredMaskContextStatistics:
        """
        Collects the structured mask statistics from the binary masks in operand.

        :return: The statistics of the structured mask context.
        """
        node = self.sparsifier_operand.target_module_node
        assert isinstance(node.layer_attributes, tuple(EXPECTED_NODE_LAYER_ATTRS))
        weight_shape: Tuple[int, int] = tuple(node.layer_attributes.get_weight_shape())
        bias_shape: Tuple[int] = (
            (node.layer_attributes.get_bias_shape(),) if self.sparsifier_operand.prune_bias else (0,)
        )

        pruned_weight_shape = list(weight_shape)
        head_id_to_keep = []
        if self.prune_by_row:
            prunable_rows = self.sparsifier_operand.weight_ctx.binary_mask.amax(dim=1)
            pruned_weight_shape[0] = int(prunable_rows.count_nonzero().item())
            kept_row_blocks = F.max_pool1d(prunable_rows.unsqueeze(0), kernel_size=self.grid_size[0]).squeeze(0)
            head_id_to_keep = kept_row_blocks.nonzero().view(-1).cpu().numpy().tolist()
        else:
            prunable_cols = self.sparsifier_operand.weight_ctx.binary_mask.amax(dim=0)
            pruned_weight_shape[1] = int(prunable_cols.count_nonzero().item())
            kept_col_blocks = F.max_pool1d(prunable_cols.unsqueeze(0), kernel_size=self.grid_size[1]).squeeze(0)
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
            module_node_name=self.module_node_name,
        )

    def _resolve_grid_size(self, grid_size) -> Tuple[int, int]:
        a, b = grid_size
        return (a if a > 0 else self.operand_mask_shape[0], b if b > 0 else self.operand_mask_shape[1])

    @staticmethod
    def _inflate_structured_mask(structured_mask: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
        assert len(structured_mask.shape) == len(
            grid_size
        ), f"Unmatched dimension with structured_mask in shape {structured_mask.shape} and grid_size in 2D."
        inflated_mask = structured_mask.clone()
        for axis, repeat_times in enumerate(grid_size):
            inflated_mask = inflated_mask.repeat_interleave(repeat_times, dim=axis)
        return inflated_mask


class StructuredMaskContextGroup:
    """
    Stores together the structured mask contexts that are related to the same building block.
    """

    def __init__(self, group_id: int, structured_mask_contexts: List[StructuredMaskContext]):
        """
        Initializes a group of related structured mask contexts.

        :param group_id: The index of the building block.
        :param structured_mask_contexts: A list of structured mask contexts corresponding
            to the building block.
        """
        self.group_id = group_id
        self.structured_mask_contexts = structured_mask_contexts

    def __str__(self) -> str:
        if not self.structured_mask_contexts:
            ctxes_str = "[]"
        else:
            ctxes = (f"\n\t{ctx}" for ctx in self.structured_mask_contexts)
            ctxes_str = "[{}\n]".format("".join(ctxes))
        return f"{self.__class__.__name__}[{self.group_id}]: {ctxes_str}"


class StructuredMaskHandler:
    """
    Handler to conduct structured masking on supported models.

    This handler gathers sparsifiable layers together as groups according to the building block
    they belong to, e.g., multi-head self-attention or feed-forward network in Transformers.
    Within each group, it refreshes the binary masks from unstructured to structured ones,
    while considering the pruning dependencies across layers. All these operations are conducted
    via the `StructuredMaskContext` of each module that supports structured masking.
    """

    def __init__(self, compressed_model: NNCFNetwork, sparsified_module_info_list: List[SparseModuleInfo]):
        """
        Initializes the handler for structured masking in movement sparsity.

        :param compressed_model: The wrapped compressed model.
        :param sparsified_module_info_list: List of `SparsifiedModuleInfo` in the
            controller of `compressed_model`.
        :param strategy: Strategy of resolving structured masks for `compressed_model`.
        """
        self.compressed_model = compressed_model
        self.sparsified_module_info_list = sparsified_module_info_list
        self._structured_mask_ctx_groups = self._create_structured_mask_context_groups(
            compressed_model, sparsified_module_info_list
        )

        nncf_logger.debug("Totally %d structured mask context groups.", len(self._structured_mask_ctx_groups))
        for structured_mask_ctx_group in self._structured_mask_ctx_groups:
            nncf_logger.debug(f"{structured_mask_ctx_group}")

    def update_independent_structured_mask(self):
        """
        Asks all contexts in `self._structured_mask_ctx_groups` to calculate the independent structured mask.
        """
        for group in self._structured_mask_ctx_groups:
            for ctx in group.structured_mask_contexts:
                ctx.update_independent_structured_mask_from_operand()

    def resolve_dependent_structured_mask(self):
        """
        Within each context group, it reads the independent structured masks of related layers and
        resolves the structured masks based on dependency rules defined in `self.rules_by_group_type`.
        """
        for group in self._structured_mask_ctx_groups:
            ctxes = group.structured_mask_contexts
            row_prune_ctxes = list(filter(lambda ctx: ctx.prune_by_row, ctxes))
            col_prune_ctxes = list(filter(lambda ctx: not ctx.prune_by_row, ctxes))
            independent_masks = [ctx.independent_structured_mask for ctx in row_prune_ctxes] + [
                ctx.independent_structured_mask.t() for ctx in col_prune_ctxes
            ]
            coarse_mask = reduce(torch.logical_or, independent_masks).float()
            with torch.no_grad():
                for ctx in row_prune_ctxes:
                    ctx.dependent_structured_mask = coarse_mask
                for ctx in col_prune_ctxes:
                    ctx.dependent_structured_mask = coarse_mask.t()

    def populate_dependent_structured_mask_to_operand(self):
        """
        Asks all contexts in `self._structured_mask_ctx_groups` to update the actual binary masks in operand.
        """
        for group in self._structured_mask_ctx_groups:
            for ctx in group.structured_mask_contexts:
                ctx.initialize_binary_mask()
        for group in self._structured_mask_ctx_groups:
            for ctx in group.structured_mask_contexts:
                ctx.populate_dependent_structured_mask_to_operand()

    def report_structured_sparsity(
        self,
        save_dir: str,
        file_name: str = "structured_sparsity",
        to_csv: bool = True,
        max_num_of_kept_heads_to_report: int = 20,
    ) -> pd.DataFrame:
        """
        Generates a report file that describes the structured mask statistics for each context group.

        :param save_dir: The folder to save the report file.
        :param file_name: File name of the report.
        :param to_csv: Whether to dump the report file in csv format.
        :param max_num_of_kept_heads_to_report: The max number of heads or channels to display that are
            preserved after structured masking. Used to avoid showing too many elements in the list.
        :return: The structured mask statistics in `pandas.DataFrame` format.
        """
        df = self._gather_statistics_dataframe(max_num_of_kept_heads_to_report)
        if to_csv:
            df.to_csv(Path(save_dir, f"{file_name}.csv"))
        return df

    def _gather_statistics_dataframe(self, max_num_of_kept_heads_to_report: int = 20) -> pd.DataFrame:
        module_vs_name_map = {module: name for name, module in self.compressed_model.named_modules()}
        entry_list = []
        for group in self._structured_mask_ctx_groups:
            ctxes = sorted(
                group.structured_mask_contexts, key=lambda ctx: ctx.sparsifier_operand.target_module_node.node_id
            )
            for ctx in ctxes:
                stats = ctx.gather_statistics_from_operand()
                module = self.compressed_model.nncf.get_containing_module(stats.module_node_name)
                entry = dict(group_id=group.group_id, torch_module=module_vs_name_map[module], **stats.__dict__)
                if len(stats.head_or_channel_id_to_keep) > max_num_of_kept_heads_to_report:  # avoid long display
                    entry["head_or_channel_id_to_keep"] = f"[{len(stats.head_or_channel_id_to_keep)} items]"
                entry_list.append(entry)
        return pd.DataFrame(entry_list)

    @staticmethod
    def _create_structured_mask_context_groups(
        nncf_network: NNCFNetwork, sparsified_module_info_list: List[SparseModuleInfo]
    ) -> List[StructuredMaskContextGroup]:
        module_vs_sparse_module_info_map = {minfo.module: minfo for minfo in sparsified_module_info_list}

        pruning_producing_types = ["linear"]
        nncf_graph = nncf_network.nncf.get_original_graph()
        pruning_groups = get_pruning_groups(
            nncf_graph, PT_EXPERIMENTAL_PRUNING_OPERATOR_METATYPES, pruning_producing_types
        )
        pruning_groups = select_largest_groups(pruning_groups)
        result = []
        for group_id, group in enumerate(pruning_groups):
            ctxes = []
            block = group.block
            extended_producers = group.producers
            for consumer in group.consumers:
                if consumer.pruning_dimension is not None:
                    extended_producers.add(ProducerInfo(consumer.node_id, consumer.pruning_dimension))
            is_group_matched = True
            prefix_warning = (
                "Automatically found structured pruning group does not match the given unstructured "
                f"pruning structures: \n{group}\n."
            )
            for producer_info in extended_producers:
                nncf_node = nncf_graph.get_node_by_id(producer_info.node_id)
                module = nncf_network.nncf.get_containing_module(nncf_node.node_name)
                if module in module_vs_sparse_module_info_map:
                    # 0 dimension corresponds to row (output channels), 1st dimension - to column (input channels)
                    prune_by_row = not bool(producer_info.pruning_dimension)
                    minfo = module_vs_sparse_module_info_map[module]
                    sparsifier: MovementSparsifier = minfo.operand
                    if sparsifier.sparse_structure == SparseStructure.PER_DIM:
                        sparse_axis = sparsifier.sparse_cfg.sparse_axis
                        if sparse_axis != producer_info.pruning_dimension:
                            nncf_logger.warning(
                                f"{prefix_warning}. Unstructured pruning is defined for "
                                f"{sparse_axis} axis, but structured one - for "
                                f"{producer_info.pruning_dimension} axis."
                            )
                            is_group_matched = False
                            break
                    prune_grid = (block.size, -1) if prune_by_row else (-1, block.size)
                    ctx = StructuredMaskContext(minfo.operand, minfo.module_node_name, prune_grid, prune_by_row)
                    ctxes.append(ctx)
                else:
                    nncf_logger.warning(
                        f"Automatically found structured pruning group does not match the given "
                        f"unstructured sparse structures:\n {group}"
                    )
                    is_group_matched = False
                    break
            if ctxes and is_group_matched:
                result.append(StructuredMaskContextGroup(group_id, ctxes))
        return result
