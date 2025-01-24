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
from enum import Enum
from functools import cmp_to_key
from functools import partial
from operator import itemgetter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import torch

from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.experimental.torch.search_building_blocks.search_graph import SearchGraph
from nncf.experimental.torch.search_building_blocks.search_graph import SearchGraphNode
from nncf.experimental.torch.search_building_blocks.search_graph import (
    check_graph_has_no_act_layer_duplication_after_block_removal,
)
from nncf.experimental.torch.search_building_blocks.search_graph import (
    check_graph_has_no_duplicate_edges_after_block_removal,
)
from nncf.experimental.torch.search_building_blocks.search_graph import (
    check_graph_has_no_hanging_edges_after_block_removal,
)
from nncf.experimental.torch.search_building_blocks.search_graph import get_num_ops_in_block
from nncf.experimental.torch.search_building_blocks.search_graph import get_search_graph
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTDropoutMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.operator_metatypes import PTMatMulMetatype
from nncf.torch.layers import NNCF_MODULES_OP_NAMES
from nncf.torch.nncf_network import NNCFNetwork

IgnoredNameOperators = [*PTDropoutMetatype.get_all_aliases(), MODEL_OUTPUT_OP_NAME]
OrdinalIDs = List[List[int]]
GroupedBlockIDs = Dict[int, List[int]]

ShapeVsNodesMap = Dict[str, Set[SearchGraphNode]]


class PotentialBuildingBlock:
    """
    Describes a building block that is uniquely defined by the first skipped node and end nodes.
    """

    def __init__(self, first_skipped_node: SearchGraphNode, end_node: SearchGraphNode):
        self.first_skipped_node = first_skipped_node
        self.end_node = end_node

    def __eq__(self, __o: "PotentialBuildingBlock") -> bool:
        return self.first_skipped_node == __o.first_skipped_node and self.end_node == __o.end_node


class BuildingBlock:
    """
    Describes a building block that is uniquely defined by the start and end nodes.
    """

    def __init__(self, start_node_name: NNCFNodeName, end_node_name: NNCFNodeName):
        self.start_node_name = start_node_name
        self.end_node_name = end_node_name

    def __eq__(self, __o: "BuildingBlock") -> bool:
        return self.start_node_name == __o.start_node_name and self.end_node_name == __o.end_node_name

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "[START NODE: {}, END_NODE: {}]".format(self.start_node_name, self.end_node_name)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {"start_node_name": self.start_node_name, "end_node_name": self.end_node_name}

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "BuildingBlock":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return BuildingBlock(**state)


class BuildingBlockType(Enum):
    """
    Describes type of building block for transformers-based network.
    `MHSA` type is characterized by the presence 4 FC and 2 MatMul layers.
    `FF` type is characterized by the presence 2 FC layers.
    """

    MHSA = "MHSA"
    FF = "FF"
    Unknown = "unknown"

    def __eq__(self, other: "BuildingBlockType"):
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        return hash(self.name)


class EBBlocksStateNames:
    BASIC_BLOCK = "basic_block"
    BLOCK_TYPE = "block_type"
    ORDINAL_IDS = "ordinal_ids"
    OP_ADDRESSES = "op_addresses"


class ExtendedBuildingBlock:
    """
    Provides extended information about building block. In addition to the addresses of boundary operations, it defines
    block type, indexes of start and end node and addresses all operations inside the block.
    """

    _state_names = EBBlocksStateNames

    def __init__(
        self,
        basic_block: BuildingBlock,
        block_type: BuildingBlockType,
        ordinal_ids: Optional[List[int]],
        op_addresses: Optional[Set[OperationAddress]],
    ):
        self.basic_block = basic_block
        self.block_type = block_type
        self.ordinal_ids = ordinal_ids
        self.op_addresses = op_addresses

    def __str__(self) -> str:
        return "[START NODE: {}, END_NODE: {}] [{}, {}] #ops={}".format(
            self.basic_block.start_node_name, self.basic_block.end_node_name, *self.ordinal_ids, len(self.op_addresses)
        )

    @property
    def start_node_name(self):
        return self.basic_block.start_node_name

    @property
    def end_node_name(self):
        return self.basic_block.end_node_name

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ExtendedBuildingBlock":
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        bbtype = BuildingBlockType(state[cls._state_names.BLOCK_TYPE])
        bblock = BuildingBlock.from_state(state[cls._state_names.BASIC_BLOCK])
        op_addresses = {
            OperationAddress.from_str(op_address_state) for op_address_state in state[cls._state_names.OP_ADDRESSES]
        }
        ordinal_ids = state[cls._state_names.ORDINAL_IDS]
        kwargs = {
            cls._state_names.BLOCK_TYPE: bbtype,
            cls._state_names.BASIC_BLOCK: bblock,
            cls._state_names.OP_ADDRESSES: op_addresses,
            cls._state_names.ORDINAL_IDS: ordinal_ids,
        }
        return cls(**kwargs)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        op_addresses = sorted([str(op_address) for op_address in self.op_addresses])
        return {
            self._state_names.BLOCK_TYPE: self.block_type.name,
            self._state_names.BASIC_BLOCK: self.basic_block.get_state(),
            self._state_names.OP_ADDRESSES: op_addresses,
            self._state_names.ORDINAL_IDS: self.ordinal_ids,
        }

    def __eq__(self, other: "ExtendedBuildingBlock"):
        return (
            self.block_type == other.block_type
            and self.basic_block == other.basic_block
            and self.op_addresses == other.op_addresses
            and self.ordinal_ids == other.ordinal_ids
        )


BuildingBlocks = List[BuildingBlock]
ExtendedBuildingBlocks = List[ExtendedBuildingBlock]


def add_node_to_aux_struct(node: SearchGraphNode, shape: List[int], shape_map: ShapeVsNodesMap):
    """
    Add to shape_map key of node for corresponds shape.
    """
    str_shape = str(shape)
    if str_shape in shape_map:
        shape_map[str_shape].add(node)
    else:
        shape_map[str_shape] = {node}


def remove_duplicates(sorted_blocks: List[PotentialBuildingBlock]) -> List[PotentialBuildingBlock]:
    """
    Removes identical blocks - that have the same nodes on boundaries
    :param sorted_blocks: a sorted list of block. Some determined order is required as soon as otherwise filtration
    may lead to a different result.
    :return: filtered list of blocks without duplicates
    """
    id_pairs = set()
    filtered_blocks = []
    for b in sorted_blocks:
        current_pair_ids = (b.first_skipped_node.main_id, b.end_node.main_id)
        if current_pair_ids not in id_pairs:
            filtered_blocks.append(b)
        id_pairs.add(current_pair_ids)
    return filtered_blocks


def compare_for_building_block(a: PotentialBuildingBlock, b: PotentialBuildingBlock):
    """
    Orders the blocks in ascending order of the end node index.
    If the indices of the end nodes are the same, the blocks are ordered by the
    index of the first skipped node.Otherwise - by number of operations in the block.
    """
    if a.end_node.bottom_id != b.end_node.bottom_id:
        return a.end_node.bottom_id - b.end_node.bottom_id
    if a.first_skipped_node.main_id != b.first_skipped_node.main_id:
        return a.first_skipped_node.main_id - b.first_skipped_node.main_id
    num_ops_a = get_num_ops_in_block(a.first_skipped_node, a.end_node)
    num_ops_b = get_num_ops_in_block(b.first_skipped_node, b.end_node)
    return num_ops_a - num_ops_b


def get_building_block_for_original_graph(
    building_blocks: List[PotentialBuildingBlock], orig_graph: PTNNCFGraph
) -> ExtendedBuildingBlocks:
    """
    Restore the original names and ids of the start and end of the block in original graph.
    Very cost expensive function, because of finding all op addresses in the block via nx.all_simple_paths function.
    """
    building_block_in_orig_format = []
    for block in building_blocks:
        id_end = block.end_node.bottom_id
        start_node_id = get_start_node_id(block, orig_graph)
        block_in_orig_format = BuildingBlock(
            orig_graph.get_node_key_by_id(start_node_id).split(" ")[-1],
            orig_graph.get_node_key_by_id(id_end).split(" ")[-1],
        )
        ordinal_ids = [start_node_id, id_end]
        op_addresses = get_all_node_op_addresses_in_block(orig_graph, block_in_orig_format)
        block_type = get_type_building_block(op_addresses)
        ext_block = ExtendedBuildingBlock(block_in_orig_format, block_type, ordinal_ids, op_addresses)
        building_block_in_orig_format.append(ext_block)
    return building_block_in_orig_format


def get_potential_candidate_for_block(search_graph: SearchGraph) -> Tuple[ShapeVsNodesMap, ShapeVsNodesMap]:
    """
    Distributes all nodes to the same output and input shapes.

    :param search_graph: A wrapper over the graph of target model, which represents the DNN execution graph transformed
    by pattern matching, merging nodes and inserting auxiliary nodes
    :return: two dictionaries that represents all input/output shapes (in a string form) of operations in the model and
    maps these shapes to nodes that have such shapes on input or output correspondingly.
    """
    act_input_shape = {}  # key - str(shape), value - set of node_keys
    act_output_shape = {}  # key - str(shape), value - set of node_keys
    for node in search_graph.get_all_nodes():
        next_edges = search_graph.get_next_edges(node.node_key)
        prev_edges = search_graph.get_prev_edges(node.node_key)
        for _, edge_attr in next_edges.items():
            search_graph.set_node_attr(
                node.node_key, SearchGraph.ACTIVATION_OUTPUT_SHAPE_ATTR, edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
            )
            if not node.is_dummy:
                add_node_to_aux_struct(node, edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR], act_output_shape)
            break
        for _, edge_attr in prev_edges.items():
            search_graph.set_node_attr(
                node.node_key, SearchGraph.ACTIVATION_OUTPUT_SHAPE_ATTR, edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]
            )
            add_node_to_aux_struct(node, edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR], act_input_shape)
            break
    return act_input_shape, act_output_shape


def itemgetter_force_tuple(*indexes):
    """
    itemgetter wrapper that always returns a tuple. The original function may return both: iterable and a single
    non-iterable element, which is not convenient in the general case.
    """
    getter = itemgetter(*indexes)
    if len(indexes) == 1:
        return lambda seq: (getter(seq),)  # Wrap in a tuple.
    return getter


def is_target_block_type(
    start_node_id: int,
    end_node_id: int,
    orig_graph: NNCFGraph,
    target_block_types: Optional[List[BuildingBlockType]] = None,
) -> bool:
    """
    Returns true if block has a type equal to one of the specified.
    :param start_node_id: bottom index of the starting node in search graph
    :param end_node_id: bottom index of the ending node in search graph
    :param orig_graph: original non-compressed graph.
    :param target_block_types: list of block types to match the type of the given block
    :return: additional info for the building blocks - block type and addresses of all operations inside the block .
    """
    if target_block_types is None:
        return True
    block_for_ops = BuildingBlock(
        orig_graph.get_node_key_by_id(start_node_id).split(" ")[-1],
        orig_graph.get_node_key_by_id(end_node_id).split(" ")[-1],
    )
    op_addresses = get_all_node_op_addresses_in_block(orig_graph, block_for_ops)
    block_type = get_type_building_block(op_addresses)
    return block_type in target_block_types


class BlockFilteringStrategy(Enum):
    """
    Defines strategy for filtering overlapping blocks.
    KEEP_SMALL - gives a preference to small blocks. It starts from the smallest block and filters all blocks that
    intersect or include it. Then it finds the next smallest block from the remaining ones and repeats the procedure.
    KEEP_SEQUENTIAL - gives a preference to sequential blocks, which follow each other in the model. This strategy
    is helpful for Progressive Shrinking Algorithm.
    """

    KEEP_SMALL = "keep_small"
    KEEP_SEQUENTIAL = "keep_sequential"


def get_building_blocks(
    compressed_model: NNCFNetwork,
    max_block_size: int = 50,
    min_block_size: int = 5,
    block_filter_strategy=BlockFilteringStrategy.KEEP_SEQUENTIAL,
    hw_fused_ops: bool = True,
    target_block_types: Optional[List[BuildingBlockType]] = None,
) -> Tuple[ExtendedBuildingBlocks, GroupedBlockIDs]:
    """
    This algorithm finds building blocks based on the analysis of the transformed graph.
    A building block is a block that satisfies the following rules:
    - has one input and one output tensors
    - input and output tensors shape are the same
    - removing a block from the graph (that is, the layers included in the block are not executed)
      does not lead to duplication of edges along which the same tensor flows
    - removing a block from the graph (that is, the layers included in the block are not executed)
      does not lead to dangling edges
    - removing a block from the graph (that is, the layers included in the block are not executed)
      does not lead to duplicate activation layers
    """
    if min_block_size > max_block_size:
        raise AttributeError(
            f"Minimal value for block size {min_block_size} can not be more than maximum one "
            f"{max_block_size}. Change max_block_size or min_block_size."
        )
    orig_graph = compressed_model.nncf.get_original_graph()  # PTNNCFGraph
    blocks = get_potential_building_blocks(orig_graph, hw_fused_ops, min_block_size, max_block_size)
    sorted_blocks = sorted(blocks, key=cmp_to_key(compare_for_building_block))
    filtered_building_blocks = remove_duplicates(sorted_blocks)
    filtered_building_blocks = remove_overlapping_blocks(
        filtered_building_blocks, block_filter_strategy, orig_graph, target_block_types
    )
    ext_building_blocks = get_building_block_for_original_graph(filtered_building_blocks, orig_graph)
    filtered_basic_blocks = [eb.basic_block for eb in ext_building_blocks]
    group_dependent = get_group_of_dependent_blocks(filtered_basic_blocks)
    return ext_building_blocks, group_dependent


def get_potential_building_blocks(orig_graph: NNCFGraph, hw_fused_ops: bool, min_block_size: int, max_block_size: int):
    """
    Builds a list of potential building blocks that has the same shapes on input and output and that satisfy some
    heuristic rules.
    :param orig_graph: NNCF graph of the non-compressed model
    :param hw_fused_ops: If True, automatic block search will not relate operations, which are fused on inference,
         into different blocks for skipping.
    :param min_block_size: minimum number of operations that should be in the block. Blocks with smaller number of
    operations are not considered.
    :param max_block_size: maximum number of operations that should be in the block. Blocks with larger number of
    operations are not considered.
    :return: list of potential building blocks
    """
    sgraph = get_search_graph(orig_graph, hw_fused_ops)
    heuristic_rules = [
        check_graph_has_no_duplicate_edges_after_block_removal,
        check_graph_has_no_act_layer_duplication_after_block_removal,
        check_graph_has_no_hanging_edges_after_block_removal,
    ]
    blocks = []
    act_input_shape, act_output_shape = get_potential_candidate_for_block(sgraph)
    for shape, first_skipped_nodes in act_input_shape.items():
        for first_skipped_node in first_skipped_nodes:
            previous_nodes = sgraph.get_prev_nodes(first_skipped_node.node_key)
            if first_skipped_node.node_type in IgnoredNameOperators or len(previous_nodes) != 1:
                continue
            for end_node in act_output_shape[shape]:
                if end_node.main_id <= first_skipped_node.main_id:
                    continue
                if end_node.node_type in IgnoredNameOperators:
                    continue
                num_ops_in_block = get_num_ops_in_block(first_skipped_node, end_node)
                if num_ops_in_block > max_block_size or num_ops_in_block < min_block_size:
                    continue
                # CHECK RULES
                all_rules_is_true = True
                for rule_fn in heuristic_rules:
                    if not rule_fn(sgraph, first_skipped_node, end_node):
                        all_rules_is_true = False
                        break
                if all_rules_is_true:
                    blocks.append(PotentialBuildingBlock(first_skipped_node, end_node))
    if len(blocks) > 300:
        nncf_logger.warning(
            "Number of potential building blocks is too much. The processing time can be high. "
            "Shallow the accepted range for the length of building blocks via "
            "max_block_size and min_block_size to accelerate the search process."
        )
    return blocks


def remove_overlapping_blocks(
    building_blocks: List[PotentialBuildingBlock],
    block_filter_strategy: BlockFilteringStrategy,
    orig_graph: NNCFGraph,
    target_block_types: List[BuildingBlockType],
) -> List[PotentialBuildingBlock]:
    """
    Applies the given filtering strategy to remove overlapping blocks - pairs of blocks that shares common operations.
    :param building_blocks: list of potential building blocks
    :param block_filter_strategy: strategy to remove overlapping blocks
    :param orig_graph: NNCF graph for non-compressed model
    :param target_block_types: list of target block types that shouldn't be removed
    :return: list of non-overlapping building blocks
    """
    start_ids = []
    end_ids = []
    num_ops_in_blocks = []
    for block in building_blocks:
        start_node_id = get_start_node_id(block, orig_graph)
        id_end = block.end_node.bottom_id
        num_ops_in_block = id_end - start_node_id - 1
        start_ids.append(start_node_id)
        end_ids.append(id_end)
        num_ops_in_blocks.append(num_ops_in_block)
    is_target_block_type_fn = partial(
        is_target_block_type, orig_graph=orig_graph, target_block_types=target_block_types
    )
    get_indexes_of_overlapping_blocks_fn = get_indexes_of_overlapping_blocks_seq
    if block_filter_strategy == BlockFilteringStrategy.KEEP_SMALL:
        get_indexes_of_overlapping_blocks_fn = get_indexes_of_overlapping_blocks_min
    ids_of_overlapping_blocks = get_indexes_of_overlapping_blocks_fn(
        start_ids=start_ids,
        end_ids=end_ids,
        num_ops_in_blocks=num_ops_in_blocks,
        is_target_block_type_fn=is_target_block_type_fn,
    )
    if ids_of_overlapping_blocks:
        building_blocks = [block for i, block in enumerate(building_blocks) if i not in ids_of_overlapping_blocks]
    return building_blocks


def get_start_node_id(block: PotentialBuildingBlock, orig_graph: NNCFGraph) -> int:
    """
    Returns id of the starting node of the block - the node right before first skipped node.
    :param block: potential building block
    :param orig_graph: original nncf graph
    :return: id of starting node
    """
    id_st = block.first_skipped_node.main_id
    first_skipped_node = orig_graph.get_node_by_id(id_st)
    input_nodes = orig_graph.get_input_nodes()
    start_node_id = id_st
    if first_skipped_node not in input_nodes and not block.first_skipped_node.is_dummy:
        previous_nodes = orig_graph.get_previous_nodes(first_skipped_node)
        num_inputs = len(previous_nodes)
        assert num_inputs == 1, f"building block should have a single input, but it has {num_inputs} inputs."
        start_node_id = previous_nodes[0].node_id
    return start_node_id


def get_indexes_of_overlapping_blocks_seq(
    start_ids: List[int],
    end_ids: List[int],
    num_ops_in_blocks: List[int],
    is_target_block_type_fn: Optional[Callable[[int, int], bool]] = None,
) -> Set[int]:
    """
    The function takes coordinates of the building block (start and end ids) and finds indexes of overlapping blocks.
    After only disjoint blocks remain after filtering the found blocks.
    :param start_ids: indexes of start node in the blocks.
    :param end_ids: indexes of end node in the blocks.
    :param num_ops_in_blocks: number of operations in the block.
    :param is_target_block_type_fn: functor that defines whether the block of target type by taking indexes of start
    and end node. Currently, types are ignored in this strategy
    :return: set of indexes of the overlapping blocks
    """
    num_blocks = len(num_ops_in_blocks)
    block_graph = nx.DiGraph()
    for i, (s, e, n) in enumerate(zip(start_ids, end_ids, num_ops_in_blocks)):
        block_graph.add_edge(s, e, attr={"cost": 4 - n, "block_id": i})

    result = set(range(num_blocks))
    ids_of_not_overlapping_blocks = set()
    while block_graph.nodes:
        ids_of_not_overlapping_nodes = nx.dag_longest_path(block_graph, weight="cost")
        node_ids_to_remove = set()
        for i in range(len(ids_of_not_overlapping_nodes) - 1):
            data = block_graph.get_edge_data(ids_of_not_overlapping_nodes[i], ids_of_not_overlapping_nodes[i + 1])
            ids_of_not_overlapping_blocks.add(data["attr"]["block_id"])
        for node_id in ids_of_not_overlapping_nodes:
            block_graph.remove_node(node_id)

        left_border = ids_of_not_overlapping_nodes[0]
        right_border = ids_of_not_overlapping_nodes[-1]
        for node_id1, node_id2, data in block_graph.edges(data=True):
            i = data["attr"]["block_id"]
            does_intersect_found_block = (
                left_border < start_ids[i] < right_border or left_border < end_ids[i] < right_border
            )
            does_include_found_block = (
                start_ids[i] <= left_border <= end_ids[i] and start_ids[i] <= right_border <= end_ids[i]
            )
            if does_intersect_found_block or does_include_found_block:
                node_ids_to_remove.add(node_id1)
                node_ids_to_remove.add(node_id2)

        for node_id in node_ids_to_remove:
            block_graph.remove_node(node_id)

    return result - ids_of_not_overlapping_blocks


def get_indexes_of_overlapping_blocks_min(
    start_ids: List[int],
    end_ids: List[int],
    num_ops_in_blocks: List[int],
    is_target_block_type_fn: Optional[Callable[[int, int], bool]] = None,
) -> Set[int]:
    """
    The function takes coordinates of the building block (start and end ids) and finds indexes of overlapping blocks.
    After only disjoint blocks remain after filtering the found blocks.
    :param start_ids: indexes of start node in the blocks.
    :param end_ids: indexes of end node in the blocks.
    :param num_ops_in_blocks: number of operations in the block.
    :param is_target_block_type_fn: functor that defines whether the block of target type by taking indexes of start
    and end node.
    :return: set of indexes of the overlapping blocks
    """
    if is_target_block_type_fn is None:
        is_target_block_type_fn = lambda *_: True
    result = set()
    if not start_ids or not end_ids or not num_ops_in_blocks:
        return result
    num_blocks = len(num_ops_in_blocks)

    indexes_to_sort, _ = zip(*sorted(enumerate(num_ops_in_blocks), key=itemgetter(1)))
    list_of_all_block_indexes = list(range(num_blocks))
    sorted_block_indexes = itemgetter_force_tuple(*indexes_to_sort)(list_of_all_block_indexes)

    curr_index = -1
    while list_of_all_block_indexes:
        is_found = False
        while not is_found and curr_index < num_blocks - 1:
            curr_index += 1
            found_block_id = sorted_block_indexes[curr_index]
            found_next_not_removed_block = found_block_id in list_of_all_block_indexes
            is_target_type = is_target_block_type_fn(start_ids[found_block_id], end_ids[found_block_id])
            if found_next_not_removed_block and is_target_type:
                is_found = True
            else:
                result.add(found_block_id)
            if found_block_id in list_of_all_block_indexes:
                list_of_all_block_indexes.remove(found_block_id)
        left_border = start_ids[found_block_id]
        right_border = end_ids[found_block_id]
        ids_to_remove = []
        for i in list_of_all_block_indexes:
            does_intersect_found_block = (
                left_border < start_ids[i] < right_border or left_border < end_ids[i] < right_border
            )
            does_include_found_block = (
                start_ids[i] <= left_border <= end_ids[i] and start_ids[i] <= right_border <= end_ids[i]
            )
            if does_intersect_found_block or does_include_found_block:
                ids_to_remove.append(i)
                result.add(i)
        for i in ids_to_remove:
            list_of_all_block_indexes.remove(i)
    return result


def get_group_of_dependent_blocks(blocks: BuildingBlocks) -> GroupedBlockIDs:
    """
    Building blocks can be categorized into groups. Blocks that follow each other in the graph
    (that is, they are connected by one edge) belong to the same group.

    :param: List of building blocks.
    :return: Dictionary where key is block index, value is group index.
    """
    if not blocks:
        return {}
    idx = 0
    groups = {idx: []}
    for i in range(len(blocks) - 1):
        next_start_node_name = blocks[i + 1].start_node_name
        curr_end_node_name = blocks[i].end_node_name
        if next_start_node_name == curr_end_node_name:
            groups[idx].append(i)
        else:
            groups[idx].append(i)
            idx += 1
            groups[idx] = []
    groups[idx].append(len(blocks) - 1)

    return groups


def get_all_node_op_addresses_in_block(graph: NNCFGraph, block: BuildingBlock) -> Set[OperationAddress]:
    """
    Returns set of operation addresses of all layers included in the block.

    :param graph: original non-compressed graph.
    :param block: Building blocks.
    :return: Set of operation addresses for building block.
    """
    simple_paths = graph.get_all_simple_paths(block.start_node_name, block.end_node_name)
    op_addresses = set()
    for node_keys_in_path in simple_paths:
        for node_key in node_keys_in_path:
            node = graph.get_node_by_key(node_key)
            op_addresses.add(OperationAddress.from_str(node.node_name))
    start_op_address = OperationAddress.from_str(block.start_node_name)
    op_addresses.remove(start_op_address)
    return op_addresses


def get_all_modules_in_blocks(
    compressed_model: NNCFNetwork, op_adresses_in_blocks: Set[OperationAddress]
) -> List[torch.nn.Module]:
    """
    Returns set of all modules included in the block.

    :param compressed_model: Target model.
    :param op_adresses_in_blocks: Set of operation addresses for building block.
    :return: List of module for building block.
    """
    modules = []
    for op_address in op_adresses_in_blocks:
        if op_address.operator_name in NNCF_MODULES_OP_NAMES:
            modules.append(compressed_model.nncf.get_module_by_scope(op_address.scope_in_model))
    return modules


def get_type_building_block(op_addresses_in_block: Set[OperationAddress]) -> BuildingBlockType:
    """
    Returns type of building block.
    """
    count_matmul = 0
    count_fc = 0
    for op_address in op_addresses_in_block:
        if op_address.operator_name in PTMatMulMetatype.get_all_aliases():
            count_matmul += 1
        if op_address.operator_name in PTLinearMetatype.get_all_aliases():
            count_fc += 1
    if count_fc == 4 and count_matmul == 2:
        return BuildingBlockType.MHSA
    if count_fc == 2 and count_matmul == 0:
        return BuildingBlockType.FF
    return BuildingBlockType.Unknown
