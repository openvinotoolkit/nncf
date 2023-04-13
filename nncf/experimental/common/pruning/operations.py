"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from collections import defaultdict
import copy
from enum import Enum
from enum import auto
from functools import reduce
from typing import List
from typing import Union
from typing import Optional
from typing import Type
from typing import Tuple
from typing import Dict

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import GetItemLayerAttributes
from nncf.common.graph.layer_attributes import ReshapeLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import MultipleOutputLayerAttributes
from nncf.common.graph.layer_attributes import PermuteLayerAttributes
from nncf.common.graph.layer_attributes import TransposeLayerAttributes
from nncf.common.logging import nncf_logger
from nncf.common.pruning.tensor_processor import NNCFPruningBaseTensorProcessor
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import identity_mask_propagation
from nncf.common.tensor import NNCFTensor
from nncf.experimental.common.pruning.nodes_grouping import PropagationGroup
from nncf.experimental.common.pruning.nodes_grouping import PropagationBlock
from nncf.experimental.common.pruning.nodes_grouping import MaskProducer
from nncf.experimental.common.pruning.nodes_grouping import PropagationMask


class BasePruningOp:
    """
    Determines meta operations which aggregate operations having common
    properties of interaction with pruning masks
    """

    subtypes = []
    additional_types = []

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        """
        Propagates the pruning mask through a node using pruning masks of all inputs and the current node (if any).

        :param node: The graph node to propagate mask through it
        :param graph: The model graph to prune
        :param tensor_processor: Interface with tensor processing methods
        """
        raise NotImplementedError

    @classmethod
    def get_all_op_aliases(cls) -> List[str]:
        """
        :return: list of all aliases of types in metatype
        """
        op_types = []
        for subtype in cls.subtypes:
            op_types.extend(subtype.get_all_aliases())
        op_types = list(set(op_types)) + cls.additional_types
        return op_types

    @classmethod
    def invalidate_masks(cls, input_masks: List[Union[None, PropagationMask]]) -> None:
        """
        Safely invalidates groups for the list of Nones and propagation masks.
        Usually such output comes from the `get_input_masks` helper.

        :param input_masks: list of input masks or N
        """
        for mask in input_masks:
            if mask is not None:
                mask.invalidate_groups()


class InputPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class OutputPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        node.data['output_mask'] = None


class IdentityMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class LinearPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) in [1, 2]
        is_none_map = map(lambda x: x is None, input_masks)
        output_mask = node.data.get('output_mask', None)
        if all(is_none_map):
            # It's acceptable to have None instead of PropagationMask on all inputs.
            # It means there's no pruning on this way to node. But the linear op can produce pruning mask,
            # need to propagate it, if it exists
            pass
        elif any(is_none_map):
            # It's NOT acceptable to have None instead of PropagationMask on one of the input and have
            # PropagationMask on another input at once. The mask is invalidated.
            cls.invalidate_masks(input_masks)
            output_mask = None
        elif len(input_masks) == 1:
            # Linear module with built-in weights. Assume single input for activations.
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            input_shape_len = len(input_tensors_shapes[0])
            # Propagate further all except the last one, which is closing
            for dim, groups in input_masks[0].dim_groups_map.items():
                if dim == input_shape_len - 1:
                    for group in groups:
                        group.close_branch()
                        cls.add_consumer_block(group, node)
                else:
                    output_mask.dim_groups_map[dim] = groups
        elif len(input_masks) == 2:
            # Matmul operator with 2 inputs: one - for activation, second - for weights.
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            assert len(input_tensors_shapes[0]) == len(input_tensors_shapes[1])
            input_shape_len = len(input_tensors_shapes[0])
            # Join consumed masks
            left_dim_groups, right_dim_groups = [input_masks[i].dim_groups_map for i in range(2)]

            def _both_dim_blocks_exist(left_idx, right_idx):
                if left_idx in left_dim_groups or \
                        right_idx in right_dim_groups:
                    assert left_idx in left_dim_groups and right_idx in right_dim_groups
                    return True
                return False

            output_mask = PropagationMask()
            # Propagating batch dims
            for dim in range(input_shape_len - 2):
                if _both_dim_blocks_exist(dim, dim):
                    left = left_dim_groups[dim]
                    right = right_dim_groups[dim]
                    assert len(left) == 1 and len(right) == 1, "multiple groups are not supported"
                    output_mask.dim_groups_map[dim] = [
                        PropagationGroup.join_groups(left_dim_groups[dim][0], right_dim_groups[dim][0])
                    ]
            # Propagating left rows / right cols
            for idx, dim in enumerate(range(input_shape_len - 2, input_shape_len)):
                if dim in input_masks[idx].dim_groups_map:
                    output_mask.dim_groups_map[dim] = input_masks[idx].dim_groups_map[dim]

            # Close branch
            if _both_dim_blocks_exist(input_shape_len - 1, input_shape_len - 2):
                left = left_dim_groups[input_shape_len - 1]
                right = right_dim_groups[input_shape_len - 2]
                assert len(left) == 1 and len(right) == 1, "multiple groups are not supported"
                group = PropagationGroup.join_groups(left[0], right[0])
                group.close_branch()
                cls.add_consumer_block(group, node)

        node.data['output_mask'] = output_mask

    @classmethod
    def add_consumer_block(cls, group: PropagationGroup, node: NNCFNode) -> None:
        """
        Explicitly adds a consumer block to the given group.

        :param group: the given block group.
        :param node: node that corresponds to the consumer.
        """
        if node.layer_attributes is not None:
            first_block: PropagationBlock = group.get_blocks()[0]
            consumer_block = PropagationBlock(
                MaskProducer(node.node_id),
                size=first_block.size,
                offset=first_block.offset,
                pruning_dimension=1,
                closed_branches=1
            )
            group.add_block(consumer_block)


class BatchNormPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        identity_mask_propagation(node, graph)


class GroupNormPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        # For Instance Normalization
        return isinstance(node.layer_attributes, GroupNormLayerAttributes) \
            and node.layer_attributes.num_groups == node.layer_attributes.num_channels

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        if cls.accept_pruned_input(node):
            identity_mask_propagation(node, graph)
        else:
            node.data['output_mask'] = None

class ElementwisePruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        if not input_masks:
            input_masks = [None]

        output_mask = input_masks[0]
        if len(input_masks) == 1:
            nncf_logger.warning(f"ElementWise with a single input is not properly supported. node_name={node.node_name}"
                                "The second input might be a constant without node in the graph. "
                                "The constant should be in the graph or in the node attributes. "
                                "It's also should be pruned in accordance with an input mask.")

        if any(m is None for m in input_masks):
            output_mask = None
            cls.invalidate_masks(input_masks)

        node.data['output_mask'] = output_mask


class GatherPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is None:
            node.data['output_mask'] = None
            return

        removed_axis = cls._is_dim_removed_by_splitting(graph, node)
        if removed_axis is not None:
            output_mask = PropagationMask()
            for input_mask in input_masks:
                for dim, groups in input_mask.dim_groups_map.items():
                    if dim != removed_axis:
                        shifted_dim = dim - 1
                        output_mask.dim_groups_map[shifted_dim] = groups
                        # other groups propagated further
            node.data['output_mask'] = output_mask
        else:
            cls.invalidate_masks(input_masks)
            node.data['output_mask'] = None

    @classmethod
    def _is_dim_removed_by_splitting(cls, graph: NNCFGraph, node: NNCFNode) -> Optional[int]:
        """
        Determines whether the operations going from parent's op is equivalent to split
        Currently, limited to the case of simple __getitem__ with axis=0 and integer key rather than more general
        `gather` operation or __getitem__ with slice or tuple.
        q, k, v = qkv[0], qkv[1], qkv[2] (like in official SwinTransformer implementation)

        - look at all consumer of parent node
        - all of them should
            - be getitem
            - slice along the same axis
            - combine the whole input (the keys should contain all dimension along axis. size of first dim=3 => [0,1,2])
        :param graph: NNCF graph to work with.
        :param node: A child node with __getitem__ operation.
        :return : axis that is removed by split, or None otherwise
        """
        split_axis = None
        if isinstance(node.layer_attributes, GetItemLayerAttributes):
            input_edge = graph.get_input_edges(node)[0]
            input_shape = input_edge.tensor_shape
            parent_node = input_edge.from_node
            child_nodes = graph.get_next_nodes(parent_node)
            child_attributes = [cnode.layer_attributes for cnode in child_nodes]
            all_getitem = all(isinstance(ca, GetItemLayerAttributes) for ca in child_attributes)
            assert all_getitem, "currently supported only case with all  __getitem__ on branches"
            all_int_keys = all(isinstance(ca.key, int) for ca in child_attributes)
            # currently supported only case __getitem__ with single int, no slices
            if not all_int_keys:
                return None
            all_keys = set(ca.key for ca in child_attributes)
            split_dim = input_shape[0]
            if all_keys == set(range(split_dim)):
                split_axis = 0
        return split_axis

class SplitPruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        if not input_masks:
            input_masks = [None]

        assert len(input_masks) == 1
        output_mask = input_masks[0]

        assert isinstance(node.layer_attributes, MultipleOutputLayerAttributes)
        chunk_axis = node.layer_attributes.axis
        chunks = node.layer_attributes.chunks

        input_edge = graph.get_input_edges(node)[0]
        input_shape = input_edge.tensor_shape
        is_chunk_axis_removed = chunks == input_shape[chunk_axis]
        if is_chunk_axis_removed:
            output_mask = PropagationMask()
            for dim, groups in input_masks[0].dim_groups_map.items():
                if dim != chunk_axis:
                    output_mask.dim_groups_map[dim] = groups
                    # other groups propagated further
        else:
            raise NotImplementedError("symbolic mask propagation for split by prune dimension is not implemented")

        node.data['output_mask'] = output_mask


class ReshapeMode(Enum):
    """
    Defines the mode of reshape.
    Here's examples of reshaping for each mode:
        Extend: [N,C,H,W] ----(reshaped to)---> [N, C1, C2, H, W1, W2], when C=C1*C2 and W=W1*W2
        Shrink is opposite to Extend: [N, C1, C2, H, W1, W2] ----(reshaped to)---> [N,C,H,W]
        "Identity without ones" happens when removing ones from input and output shapes leads to identity mapping.
            For instance: [1, N] -> [N] is [N] or [N, 1, C] -> [1, N, 1, 1, C]
        Default - all other cases.
    """
    SHRINK = auto()
    EXTEND = auto()
    DEFAULT = auto()
    IDENTITY_WITHOUT_ONES = auto()


class ReshapePruningOp(BasePruningOp):
    DIMENSION_MAP = Dict[int, List[int]]

    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        if node.layer_attributes is None:
            return False
        return True

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        masks = get_input_masks(node, graph)
        assert len(masks) == 1
        mask = masks[0]
        output_mask = PropagationMask()
        if mask is not None and node.layer_attributes:
            in_map, _, mode = cls.parse_reshape(node.layer_attributes)
            input_shape = node.layer_attributes.input_shape
            output_shape = node.layer_attributes.output_shape

            if mode == ReshapeMode.DEFAULT:
                nncf_logger.warning(f'Detected not supported reshape from {input_shape} to {output_shape}. '
                                    f'The propagation mask is invalidated for node={node.node_name}.')
                cls.invalidate_masks(masks)
                output_mask = None
            elif mode == ReshapeMode.IDENTITY_WITHOUT_ONES:
                # pruning dimension is not shrunk and extended, just assign a new location in the output
                for dim, groups in mask.dim_groups_map.items():
                    assert len(in_map[dim]) == 1, "assume a mapping to single int by definition of identity"
                    shifted_dim = in_map[dim][0]
                    output_mask.dim_groups_map[shifted_dim] = groups
            elif mode == ReshapeMode.EXTEND:
                for dim, groups in mask.dim_groups_map.items():
                    if len(groups) > 1:
                        raise NotImplementedError('Extend reshape for several groups is not supported yet')
                    if len(in_map[dim]) == 1:
                        # pruning dimension is not extended, just assign a new location in the output
                        shifted_dim = in_map[dim][0]
                        output_mask.dim_groups_map[shifted_dim] = groups
                    else:
                        # extension like this: S -> [A,B,C,D]. Need to split groups.
                        input_channels = input_shape[dim]
                        list_output_channels = [output_shape[x] for x in in_map[dim]]
                        group = groups[0]
                        if group.has_children():
                            raise NotImplementedError('Splitting BlockGroup with children is not implemented yet')
                        child_groups = cls._split_group(group, input_channels, list_output_channels)
                        for child_group, in_dim in zip(child_groups, in_map[dim]):
                            output_mask.dim_groups_map[in_dim] = [child_group]
            elif mode == ReshapeMode.SHRINK:
                # shrinking like: [A,B,C,D] -> S, just combine all groups under the same dimension
                grouping = defaultdict(list)
                for in_idx, groups in mask.dim_groups_map.items():
                    assert len(in_map[in_idx]) == 1, "assume a mapping to single int by definition of shrink"
                    grouping[in_map[in_idx][0]].extend(groups)
                output_mask.dim_groups_map = dict(grouping)

        node.data['output_mask'] = output_mask

    @staticmethod
    def _can_reach_number_by_multiply(number_to_reach: int,
                                      array: List[int],
                                      start_idx: int) -> Tuple[bool, int]:
        """
        Sequentially multiplies elements in the given array starting from the given start index until reaching or
        exceeding the given number.

        :param number_to_reach: number to reach while multiplying elements.
        :param array: array of numbers.
        :param start_idx: the index, starting from which the numbers in the given array will be sequentially multiplied.
        :return: a pair, when first element is True when the given number is reached, False - otherwise.
            The second element is the index of last element in the array when the product of numbers reached or
            exceeded the given number.
        """
        idx = start_idx
        accum = array[idx]
        while accum < number_to_reach:
            idx += 1
            accum *= array[idx]
        if accum > number_to_reach:
            return (False, idx)
        return (True, idx)

    @classmethod
    def _map_dims_(cls,
                   source_array: List[int],
                   target_array: List[int],
                   source_idx: int,
                   start_target_idx: int,
                   source_to_target_map: DIMENSION_MAP,
                   target_to_source_map: DIMENSION_MAP) -> Tuple[bool, int]:
        """
        The function tries to decompose one element in the source array by the product of sequential elements in the
        target array. If it can be decomposed it's considered "mapped". i.e. the source element corresponds to the list
        of elements in the target array, end vice-versa, elements in the target array corresponds to the element in the
        source array. By this correspondence, mappings of element's indexes from source to target and from target to
        source are built.

        :param source_array: array to get a single element for mapping to elements in the target array.
        :param target_array: array to find element(s) that map(s) to a single element in the source array.
        :param source_idx: index of the single element in the source array.
        :param target_start_idx: index in the target array to start sequentially multiplying elements until reaching
            the element from source array.
        :param source_to_target_map: mapping of indexes of source elements to indexes of target elements.
        :param target_to_source_map: mapping of indexes of target elements to indexes of source elements.
        :return: a pair, when first element is True when the element from source array can be mapped to sequential
            element(s) of target array, False - otherwise.
            The second element of pair is the index of last mapped element in the target array.
        """
        res, last_target_index = cls._can_reach_number_by_multiply(
            number_to_reach=source_array[source_idx],
            array=target_array,
            start_idx=start_target_idx
        )
        if not res:
            return (res, last_target_index)
        source_to_target_map[source_idx] = list(range(start_target_idx, last_target_index + 1))
        for idx in range(start_target_idx, last_target_index + 1):
            target_to_source_map[idx] = [source_idx]
        return (res, last_target_index)

    @classmethod
    def parse_reshape(cls,
                      reshape_attributes: ReshapeLayerAttributes) -> Tuple[DIMENSION_MAP, DIMENSION_MAP, ReshapeMode]:
        """
        Parse reshape semantic by mapping elements in the input shape to elements of output shape.

        :param reshape_attributes: attributes that contains input and output shape.
        :return: return two mappings (from input to output, and vice versa) and reshape mode.
        """
        input_shape_not_cut = reshape_attributes.input_shape
        output_shape_not_cut = reshape_attributes.output_shape

        in_indexes_not_cut_map = [i for i, dim in enumerate(input_shape_not_cut) if dim != 1]
        out_indexes_not_cut_map = [i for i, dim in enumerate(output_shape_not_cut) if dim != 1]

        input_shape=list(filter(lambda x: x != 1, input_shape_not_cut))
        output_shape=list(filter(lambda x: x != 1, output_shape_not_cut))

        in_idx = 0
        out_idx = 0
        in_map = {}
        out_map = {}

        mode = ReshapeMode.DEFAULT

        while (in_idx < len(input_shape) and out_idx < len(output_shape)):
            if input_shape[in_idx] == output_shape[out_idx]:
                in_map[in_idx] = [out_idx]
                out_map[out_idx] = [in_idx]
            elif input_shape[in_idx] > output_shape[out_idx]:
                res, out_idx = cls._map_dims_(
                    source_array=input_shape,
                    target_array=output_shape,
                    source_idx=in_idx,
                    start_target_idx=out_idx,
                    source_to_target_map=in_map,
                    target_to_source_map=out_map
                )
                if not res or mode == ReshapeMode.SHRINK:
                    return None, None, ReshapeMode.DEFAULT
                mode = ReshapeMode.EXTEND
            else:
                res, in_idx = cls._map_dims_(
                    source_array=output_shape,
                    target_array=input_shape,
                    source_idx=out_idx,
                    start_target_idx=in_idx,
                    source_to_target_map=out_map,
                    target_to_source_map=in_map
                )
                if not res or mode == ReshapeMode.EXTEND:
                    return None, None, ReshapeMode.DEFAULT
                mode = ReshapeMode.SHRINK
            in_idx += 1
            out_idx += 1

        if mode == ReshapeMode.DEFAULT:
            mode = ReshapeMode.IDENTITY_WITHOUT_ONES

        in_map_not_cut = cls._convert_to_not_cut(in_indexes_not_cut_map, out_indexes_not_cut_map, in_map)
        out_map_not_cut = cls._convert_to_not_cut(out_indexes_not_cut_map, in_indexes_not_cut_map, out_map)
        return in_map_not_cut, out_map_not_cut, mode

    @classmethod
    def _convert_to_not_cut(cls,
                            source_indexes_not_cut_map: List[int],
                            target_indexes_not_cut_map: List[int],
                            source_to_targets_map: DIMENSION_MAP
        ) -> DIMENSION_MAP:
        """
        Perform re-mapping indexes for the cut arrays to non-cut ones.

        :param source_indexes_not_cut_map: indexes of not cut elements in the uncut source array.
            The indexes are in the order of the elements in the array, so it can be used to map indexes from a cut array
            to uncut one.
        :param target_indexes_not_cut_map: indexes of not cut elements in the uncut target array. The indexes are in the
            order of the elements in the array, so it can be used to map indexes from a cut array to uncut one.
        :param source_to_targets_map: mapping from source elements to target ones in the cut arrays.
        :return: mapping from source elements to target ones in the not cut arrays.
        """
        source_to_targets_map_not_cut = {}
        for source_index, target_indexes in source_to_targets_map.items():
            source_index_not_cut = source_indexes_not_cut_map[source_index]
            target_indexes_not_cut = list(map(lambda x: target_indexes_not_cut_map[x], target_indexes))
            source_to_targets_map_not_cut[source_index_not_cut] = target_indexes_not_cut
        return source_to_targets_map_not_cut

    @staticmethod
    def _split_block(
        block: PropagationBlock,
        list_output_channels: List[int]
    ) -> List[PropagationBlock]:
        """
        Splits a dimension block into multiple blocks.
        It's applied when some number of channels S is reshaped to N channels, e.g. S -> [A,B,C,D] and S=A*B*C*D.
        Now we assume that pruning is possible along each new dimension instead of S one.
        The dimension block encoding pruning of a single channel from S is no longer valid.
        This function creates new blocks that encodes how many channels from S is pruned if we prune along a new
        dimension from [A,B,C,D].
        It forms new constraints by the following rules:
            | size  | offset          |
            |-------|-----------------|
            | 1     | D % S           |
            | D     | C*D % S         |
            | C*D   | B*C*D % S       |
            | B*C*D | A*B*C*D % S = 0 |

        :param block: dimension block to split
        :param list_output_channels: list of output channels (A,B,C,D)
        :return:
        """
        if len(list_output_channels) == 1:
            raise RuntimeError

        dot_product = reduce((lambda x, y: x * y), list_output_channels)

        current_size = dot_product
        new_blocks = []
        divided_shapes = filter(lambda x: x != 1, list_output_channels)
        for divided_shape in divided_shapes:
            offset = current_size % dot_product
            current_size /= divided_shape
            new_block = copy.copy(block)
            new_block.size = int(current_size)
            new_block.offset = offset
            new_blocks.append(new_block)
        return new_blocks

    @classmethod
    def _split_group(cls,
                     group: PropagationGroup,
                     input_channels: int,
                     list_output_channels: List[int]) -> List[PropagationGroup]:
        """
        Splits a group of dimension blocks into new child groups by a pruning dimension.
        Internally calls "_split_block" to split each block from the given group. Please refer to the
        description of the method for more details about splitting blocks.
        Group can have multiple blocks. E.g. s1 and s2. Let's say s1 is splitted into (a1,b1), s2 - into (a2,b2).
        The method will return 2 children groups: (a1,a2) and (b1,b2).

        :param input_channels: splitted input channels (S).
        :param list_output_channels: list of output channels (A,B,C,D).
        :param group: a group of dimension blocks for splitting.
        :return: list of new groups which results from the split.
        """
        new_blocks: List[List[PropagationBlock]] = []
        child_groups: List[PropagationGroup] = []

        for block in group.get_blocks():
            dot_product = reduce((lambda x, y: x * y), list_output_channels)
            assert dot_product == input_channels
            new_blocks.append(cls._split_block(block, list_output_channels))

        # forms [(a1,a2), (b1,b2)] from [(a1,b1), (a2,b2)]
        for block_groups in zip(*new_blocks):
            child_group = PropagationGroup(blocks=list(block_groups))
            group.add_child(child_group)
            child_groups.append(child_group)
        return child_groups


class TransposePruningOp(BasePruningOp):
    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        assert len(input_masks) == 1
        input_mask = input_masks[0]
        if input_mask is None:
            node.data['output_mask'] = None
            return

        if isinstance(node.layer_attributes, TransposeLayerAttributes):
            input_tensors_shapes = [x.tensor_shape for x in graph.get_input_edges(node)]
            assert len(input_tensors_shapes) == 1
            new_order = list(range(len(input_tensors_shapes[0])))
            dim0 = node.layer_attributes.dim0
            dim1 = node.layer_attributes.dim1
            new_order[dim1], new_order[dim0] = new_order[dim0], new_order[dim1]
        elif isinstance(node.layer_attributes, PermuteLayerAttributes):
            new_order = node.layer_attributes.permutation

        idx_map = [(old_idx, new_idx) for new_idx, old_idx in enumerate(new_order) if
                   old_idx in input_mask.dim_groups_map]
        output_mask = PropagationMask(
            dim_groups_map={
                new_idx: input_mask.dim_groups_map[old_idx] for old_idx, new_idx in idx_map
            }
        )

        node.data['output_mask'] = output_mask


class StopMaskForwardPruningOp(BasePruningOp):
    @classmethod
    def accept_pruned_input(cls, node: NNCFNode) -> bool:
        return False

    @classmethod
    def mask_propagation(cls, node: NNCFNode, graph: NNCFGraph,
                         tensor_processor: Type[NNCFPruningBaseTensorProcessor]) -> None:
        input_masks = get_input_masks(node, graph)
        cls.invalidate_masks(input_masks)
        node.data['output_mask'] = None
