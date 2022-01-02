"""
 Copyright (c) 2021 Intel Corporation
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
from collections import deque
from copy import deepcopy
from enum import Enum
from functools import cmp_to_key
from itertools import combinations
from itertools import groupby
from typing import Any, List, Set, Tuple, Dict

import networkx as nx
import torch

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.hardware.fused_patterns import PT_HW_FUSED_PATTERNS
from nncf.torch.graph.operator_metatypes import PTDropoutMetatype
from nncf.torch.graph.operator_metatypes import PTRELUMetatype
from nncf.torch.graph.operator_metatypes import PTMatMulMetatype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.layers import NNCF_MODULES_OP_NAMES
from nncf.torch.nncf_network import NNCFNetwork

IGNORED_NAME_OPERATORS = [*PTDropoutMetatype.get_all_aliases(), MODEL_OUTPUT_OP_NAME]

bar = 42

class SearchGraphNode:
    """
    Class describing nodes used in SearchGraph.
    """
    def __init__(self,
                 node_key: str,
                 data: Dict = None):
        self.node_key = node_key
        self.data = data if data else {}

    @property
    def is_merged(self) -> bool:
        """
        Returns value that the node is merged.
        """
        return self.data.get(SearchGraph.IS_MERGED_NODE_ATTR)

    @property
    def is_dummy(self) -> bool:
        """
        Returns value that the node is dummy.
        """
        return self.data.get(SearchGraph.IS_DUMMY_NODE_ATTR)

    @property
    def node_name(self) -> NNCFNodeName:
        return self.data.get(NNCFGraph.NODE_NAME_ATTR)

    @property
    def node_type(self) -> str:
        """
        Returns type of node.
        """
        return self.data.get(NNCFGraph.NODE_TYPE_ATTR)

    @property
    def layer_name(self) -> str:
        """
        Returns the name of the layer to which the node corresponds.
        """
        return self.data.get(NNCFGraph.LAYER_NAME_ATTR)

    @property
    def main_id(self) -> int:
        """
        Returns the id of node. In case if the node is merged returns id of the first node from merged node list.
        """
        if not self.is_merged:
            return self.data.get('id')
        return self.data.get(SearchGraph.MERGED_NODES_NODE_ATTR)[0].get('id')

    @property
    def bottom_id(self) -> int:
        """
        Returns the id of node. In case if the node is merged returns id of the last node from merged node list.
        """
        if not self.is_merged:
            return self.data.get('id')
        return self.data.get(SearchGraph.MERGED_NODES_NODE_ATTR)[-1].get('id')

    def __str__(self):
        return self.node_key

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, SearchGraphNode) and self.node_key == other.node_key


class BuildingBlock:
    """
    Describes a building block that is uniquely defined by the start and end nodes.
    """
    def __init__(self, start_node: SearchGraphNode, end_node: SearchGraphNode):
        self.start_node = start_node
        self.end_node = end_node

    def __eq__(self, __o: object) -> bool:
        return self.start_node == __o.start_node and self.end_node == __o.end_node


class BuildingBlockType(Enum):
    """
    Describes type of building block for transformers-based network.
    `MSHA` type is characterized by the presence 4 FC and 2 MatMul layers.
    `FF` type is characterized by the presence 2 FC layers.
    """
    MSHA = 'MSHA'
    FF = 'FF'
    Unknown = 'unknown'


class BuildingBlockInfo:
    """
    Describes additional information about the building block
    the address of each layer, the modules contained and type of block.
    """
    def __init__(self, building_block: BuildingBlock,
                       op_addresses: List[OperationAddress],
                       modules: List[torch.nn.Module],
                       block_type: BuildingBlockType):
        self.building_block = building_block
        self.op_addresses = op_addresses
        self.modules = modules
        self.block_type = block_type


class SearchGraph:
    """
    A wrapper over the graph, which represents the DNN execution graph transformed
    by pattern matching, merging nodes and inserting auxiliary nodes.
    """
    ACTIVATION_INPUT_SHAPE_ATTR = 'activation_input_shape'
    ACTIVATION_OUTPUT_SHAPE_ATTR = 'activation_output_shape'
    IS_MERGED_NODE_ATTR = 'is_merged'
    IS_DUMMY_NODE_ATTR = 'is_dummy'
    KEY_NODE_ATTR = 'key'
    MERGED_NODES_NODE_ATTR = 'merged_nodes'
    TYPE_NODE_ATTR = 'type'
    DUMMY_POSTFIX = " dummy"


    def __init__(self, nx_merged_graph: nx.DiGraph):
        self._nx_graph = nx_merged_graph
        old_merged_graph_nodes = deepcopy(self._nx_graph._node)
        for node_key in old_merged_graph_nodes:
            # pylint: disable=protected-access
            next_nodes = self._nx_graph._succ[node_key]
            if len(list(next_nodes)) > 1:
                self._insert_dummy_node(node_key)

    def _nx_node_to_sgraph_node(self, nx_node_key: str, nx_node_attrs: Dict[str, Any]):
        return SearchGraphNode(nx_node_key, nx_node_attrs)

    def get_node_by_key(self, node_key: str) -> SearchGraphNode:
        """
        :param node_key: key (node_name) of the node.
        :return: SearchGraphNode in a graph with such key.
        """
        return SearchGraphNode(node_key, self._nx_graph.nodes[node_key])

    def get_all_nodes(self) -> List[SearchGraphNode]:
        """
        Returns list of all graph nodes.
        """
        all_nodes = []
        for node_key, node_attrs in self._nx_graph.nodes.items():
            all_nodes.append(SearchGraphNode(node_key, node_attrs))
        return all_nodes

    def get_next_nodes(self, node_key: str) -> List[SearchGraphNode]:
        """
        Returns consumer nodes of provided node key.

        :param node_key: Producer node key.
        :return: List of consumer nodes of provided node.
        """
        next_node_keys = self._nx_graph.succ[node_key]
        return [self.get_node_by_key(node_key) for node_key in next_node_keys]

    def set_node_attr(self, node_key: str, name_attr: str, value_attr: str):
        """
        Set value of attribute by name for a given node with the same key.
        """
        self._nx_graph.nodes[node_key][name_attr] = value_attr

    def get_prev_nodes(self, node_key: str) -> List[SearchGraphNode]:
        """
        Returns producer nodes of provided node key.

        :param node_key: Consumer node key.
        :return: List of producers nodes of provided node.
        """
        prev_node_keys = self._nx_graph.pred[node_key]
        return [self.get_node_by_key(node_key) for node_key in prev_node_keys]

    def get_prev_edges(self, node_key: str) -> Dict[str, Any]:
        return self._nx_graph.pred[node_key]

    def get_next_edges(self, node_key: str) -> Dict[str, Any]:
        return self._nx_graph.succ[node_key]

    def _insert_dummy_node(self, node_key: str):
        # pylint: disable=protected-access
        next_nodes = deepcopy(self._nx_graph._succ[node_key])
        dummy_node_attr = {SearchGraph.IS_DUMMY_NODE_ATTR: True}
        node_key_attr = deepcopy(self._nx_graph._node[node_key])
        dummy_node_key = node_key + SearchGraph.DUMMY_POSTFIX
        node_key_attr.update(dummy_node_attr)
        self._nx_graph.add_node(dummy_node_key, **node_key_attr)

        edge_attrs = None
        for next_node_key in next_nodes:
            edge_attrs = self._nx_graph.get_edge_data(node_key, next_node_key)
            self._nx_graph.remove_edge(node_key, next_node_key)
            self._nx_graph.add_edge(dummy_node_key, next_node_key, **edge_attrs)
        self._nx_graph.add_edge(node_key, dummy_node_key, **edge_attrs)

    def get_nx_graph(self) -> nx.DiGraph:
        """
        Returns internal representation of SearchGraph as a networkx graph.
        """
        return self._nx_graph


def get_search_graph(original_graph: PTNNCFGraph) -> SearchGraph:
    """
    Returns a transformed representation of the network graph for blocks searching.
    """
    nx_merged_graph = get_merged_original_graph_with_pattern(original_graph.get_nx_graph_copy())
    sgraph = SearchGraph(nx_merged_graph)
    return sgraph


def get_merged_original_graph_with_pattern(orig_graph: nx.DiGraph) -> nx.DiGraph:
    """
    :param orig_graph: Original graph of model
    :return: Graph with merged nodes by patterns
    """
    # pylint: disable=protected-access
    pattern_fusing_graph = PT_HW_FUSED_PATTERNS.get_full_pattern_graph()
    matches = find_subgraphs_matching_pattern(orig_graph, pattern_fusing_graph)
    merged_graph = deepcopy(orig_graph)
    nx.set_node_attributes(merged_graph, False, SearchGraph.IS_DUMMY_NODE_ATTR)
    nx.set_node_attributes(merged_graph, False, SearchGraph.IS_MERGED_NODE_ATTR)
    nx.set_node_attributes(merged_graph, None, SearchGraph.ACTIVATION_INPUT_SHAPE_ATTR)
    nx.set_node_attributes(merged_graph, None, SearchGraph.ACTIVATION_OUTPUT_SHAPE_ATTR)
    for match in matches:
        if len(match) == 1:
            continue
        input_node_key = match[0]
        output_node_key = match[-1]
        in_edges = list(merged_graph.in_edges(input_node_key))
        out_edges = list(merged_graph.out_edges(output_node_key))

        in_edge_copies_dict = {}
        for in_edge_key in in_edges:
            in_edge_copies_dict[in_edge_key] = deepcopy(merged_graph.edges[in_edge_key])
        out_edge_copies_dict = {}
        for out_edge_key in out_edges:
            out_edge_copies_dict[out_edge_key] = deepcopy(merged_graph.edges[out_edge_key])

        merged_node_key = ""
        merged_nodes = []
        type_list = []
        for node_key in match:
            attrs = orig_graph.nodes[node_key]
            merged_node_key +=  str(attrs['id']) + ' ' + attrs[SearchGraph.TYPE_NODE_ATTR] + '  '
            # pylint: disable=protected-access
            merged_nodes.append(orig_graph.nodes[node_key])
            merged_graph.remove_node(node_key)
            type_list.append(attrs[SearchGraph.TYPE_NODE_ATTR])
        merged_node_attrs = {
            SearchGraph.KEY_NODE_ATTR: merged_node_key,
            SearchGraph.IS_MERGED_NODE_ATTR: True,
            SearchGraph.TYPE_NODE_ATTR: type_list,
            SearchGraph.MERGED_NODES_NODE_ATTR: merged_nodes,
        }
        merged_graph.add_node(merged_node_key, **merged_node_attrs)
        for in_edge_key, in_edge_attrs in in_edge_copies_dict.items():
            merged_graph.add_edge(in_edge_key[0], merged_node_key, **in_edge_attrs)
        for out_edge_key, out_edge_attrs in out_edge_copies_dict.items():
            merged_graph.add_edge(merged_node_key, out_edge_key[1], **out_edge_attrs)

    return merged_graph


def add_node_to_aux_struct(node_key: str, shape: List[int], shape_map: Dict[str, Set[str]]):
    """
    Add to shape_map key of node for corresponds shape.
    """
    str_shape = str(shape)
    if str_shape in shape_map:
        shape_map[str_shape].add(node_key)
    else:
        shape_map[str_shape] = set([node_key])


def check_graph_has_no_hanging_edges_after_block_removal(graph: SearchGraph,
                                                         start_node: SearchGraphNode,
                                                         end_node: SearchGraphNode) -> bool:
    """
    The subgraph is traversed starting with the start_node and ending with the end_node
    to determine that after deleting such a block there are no dangling edges in the graph.
    """
    #            A
    #           / \
    #          /   \
    #         /     \
    #    start_node  |   Edge A-C is dangling edges
    #        |       |
    #         \     /
    #          \   /
    #            C
    #            |
    #           ...
    #            |
    #         end_node

    q = deque([start_node])
    addit_nodes = set()
    nodes = []
    current_node = start_node
    potential_end_nodes = []
    while len(q) != 0:
        current_node = q.pop()
        if current_node.main_id != start_node.main_id:
            prev_nodes = graph.get_prev_nodes(current_node.node_key)
            if len(prev_nodes) > 1:
                for pn in prev_nodes:
                    if pn.bottom_id < start_node.bottom_id:
                        return False # there is extra edge
                    addit_nodes.add(pn)
        if current_node.node_key == end_node.node_key:
            continue
        if current_node.main_id > end_node.main_id:
            return False
        if current_node not in nodes:
            nodes.append(current_node)
            next_nodes = graph.get_next_nodes(current_node.node_key)
            if len(next_nodes) == 0:
                potential_end_nodes.append(current_node)
            for next_node in next_nodes:
                q.appendleft(next_node)
    if len(q) > 0 or len(potential_end_nodes) > 0:
        return False
    for node in addit_nodes:
        if node not in nodes:
            return False
    nodes.append(end_node)
    return True


def check_graph_has_no_duplicate_edges_after_block_removal(sgraph: SearchGraph,
                                                           start_node: SearchGraphNode,
                                                           end_node: SearchGraphNode) -> bool:
    """
    This rule ensures that no duplicate edges will be created in the graph after a block is deleted.
    """
    #         A              A
    #        / \            / \
    #       /   \          /   \
    #    block   |  =>     \   /
    #       \   /           \ /
    #        \ /             D
    #         D          forbidden

    #pylint: disable=protected-access
    if start_node.is_dummy:
        next_end_node = sgraph.get_next_nodes(end_node.node_key)
        if len(next_end_node) != 0:
            attr = sgraph._nx_graph.get_edge_data(start_node.node_key, next_end_node[0].node_key)
        else:
            attr = None
    else:
        pred_start_node = sgraph.get_prev_nodes(start_node.node_key)
        next_end_node = sgraph.get_next_nodes(end_node.node_key)
        if len(pred_start_node) != 0 and len(next_end_node) != 0:
            attr = sgraph._nx_graph.get_edge_data(pred_start_node[0].node_key, next_end_node[0].node_key)
        else:
            attr = None
    return attr is None


def check_graph_has_no_act_layer_duplication_after_block_removal(sgraph: SearchGraph,
                                                                 start_node: SearchGraphNode,
                                                                 end_node: SearchGraphNode) -> bool:
    """
    This rule ensures that after the block is deleted there will be no duplication of activation layers.
    """
    #         A             A
    #         |             |
    #        relu          relu
    #         |      =>     |
    #        block         relu
    #         |
    #        relu        forbidden

    pred_start_node = sgraph.get_prev_nodes(start_node.node_key)
    next_end_node = sgraph.get_next_nodes(end_node.node_key)
    if len(next_end_node) == 0 or len(pred_start_node) == 0:
        return True
    if pred_start_node[0].is_dummy:
        pred_start_node = sgraph.get_prev_nodes(pred_start_node[0].node_key)

    if pred_start_node[0].node_type[-1] in PTRELUMetatype.get_all_aliases()\
         and next_end_node[0].node_type[0] in PTRELUMetatype.get_all_aliases():
        return False
    return True


def compare_for_building_block(a: BuildingBlock, b: BuildingBlock):
    """
    Orders the blocks in ascending order of the end node index.
    If the indices of the end nodes are the same, the blocks are ordered by the
    index of the start node.
    """
    if a.end_node.bottom_id != b.end_node.bottom_id:
        return a.end_node.bottom_id - b.end_node.bottom_id
    return b.start_node.main_id - a.start_node.main_id


def check_blocks_combination_is_block(block: BuildingBlock, combination: List[BuildingBlock]) -> bool:
    """
    Checks that a combination of blocks is a given block.
    """
    if block.start_node.main_id != combination[0].start_node.main_id:
        return False
    if block.end_node.bottom_id != combination[-1].end_node.bottom_id:
        return False
    i = 0
    while i < len(combination) - 1:
        end_i_node = combination[i].end_node
        start_nexti_node = combination[i + 1].start_node
        if end_i_node.node_key in start_nexti_node.node_key:
            i += 1
            continue
        return False
    return True


def search_lin_combination(block: BuildingBlock, blocks: List[BuildingBlock]) -> bool:
    """
    Checks that a given block is linear combination of some blocks.
    A linear combination of blocks is a sequence of blocks following each other in the graph
    and connected by one edge.
    """
    max_num = len(blocks)
    all_combinations = []
    for i in range(max_num, 1, -1):
        all_combinations = list(combinations(blocks, i))
        for combo in all_combinations:
            if check_blocks_combination_is_block(block, combo):
                return True
    return False


def remove_linear_combination(sorted_building_blocks: List[BuildingBlock]) -> List[BuildingBlock]:
    """
    Search and remove of block which is a combination of other blocks following each other.
    """
    result_blocks = []
    start_to_idx = {}
    last_node = None
    for block in sorted_building_blocks:
        if last_node == block.end_node:
            if block.start_node.main_id in start_to_idx:
                if not search_lin_combination(block, result_blocks[start_to_idx[block.start_node.main_id]:]):
                    result_blocks.append(block)
                    last_node = block.end_node
        else:
            result_blocks.append(block)
            last_node = block.end_node
            if block.start_node.main_id not in start_to_idx:
                start_to_idx[block.start_node.main_id] = len(result_blocks) - 1

    return result_blocks


def restore_node_name_in_orig_graph(building_blocks: List[BuildingBlock], orig_graph: PTNNCFGraph) -> List[str]:
    """
    Restore the original names of the start and end of the block in original graph.
    """
    building_block_in_orig_format = []
    for block in building_blocks:
        id_st = block.start_node.bottom_id # dummy node
        id_end = block.end_node.bottom_id
        block_in_orig_format = BuildingBlock(orig_graph.get_node_key_by_id(id_st).split(' ')[-1],
                                orig_graph.get_node_key_by_id(id_end).split(' ')[-1])
        building_block_in_orig_format.append(block_in_orig_format)
    return building_block_in_orig_format


def get_potential_candidate_for_block(sgraph: SearchGraph) -> Tuple[Dict[str, List[int]]]:
    """
    Distributes all nodes to the same output and input shapes.

    param: sgraph: SeacrhGraph of target model
    returns: Dict for input/output shapes, where key - shape,
    value - list of node with such input/output shape.
    """
    act_input_shape = {} # key - str(shape), value - set of node_keys
    act_output_shape = {} # key - str(shape), value - set of node_keys
    for node in sgraph.get_all_nodes():
        next_edges = sgraph.get_next_edges(node.node_key)
        prev_edges = sgraph.get_prev_edges(node.node_key)
        for _, edge_attr in next_edges.items():
            sgraph.set_node_attr(node.node_key, SearchGraph.ACTIVATION_OUTPUT_SHAPE_ATTR,
             edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            if not node.is_dummy:
                add_node_to_aux_struct(node, edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR], act_output_shape)
            break
        for _, edge_attr in prev_edges.items():
            sgraph.set_node_attr(node.node_key, SearchGraph.ACTIVATION_OUTPUT_SHAPE_ATTR,
             edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            break
        add_node_to_aux_struct(node, edge_attr[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR], act_input_shape)
    return act_input_shape, act_output_shape


def get_building_blocks(compressed_model: NNCFNetwork,
                        max_block_size: int = 50,
                        allow_nested_blocks: bool = True,
                        allow_linear_combination: bool = False) -> List[BuildingBlock]:

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

    orig_graph = compressed_model.get_original_graph() # PTNNCFGraph
    sgraph = get_search_graph(orig_graph)

    fn_rules = [check_graph_has_no_duplicate_edges_after_block_removal,
                check_graph_has_no_act_layer_duplication_after_block_removal,
                check_graph_has_no_hanging_edges_after_block_removal]

    blocks = []
    act_input_shape, act_output_shape = get_potential_candidate_for_block(sgraph)

    for shape, start_nodes in act_input_shape.items():
        for start_node in start_nodes:
            pred_start_node = sgraph.get_prev_nodes(start_node.node_key)
            if start_node.node_type == IGNORED_NAME_OPERATORS or len(pred_start_node) != 1:
                continue
            for end_node in act_output_shape[shape]:
                if end_node.main_id - start_node.main_id > max_block_size:
                    continue
                if end_node.node_type in IGNORED_NAME_OPERATORS:
                    continue
                if end_node.main_id <= start_node.main_id:
                    continue
                is_one_edge = (end_node.main_id - start_node.bottom_id) == 1

                # CHECK RULES
                all_rules_is_true = True
                for rule_fn in fn_rules:
                    if not rule_fn(sgraph, start_node, end_node):
                        all_rules_is_true = False
                        break
                if all_rules_is_true and not is_one_edge:
                    blocks.append(BuildingBlock(start_node, end_node))

    sorted_blocks = sorted(blocks, key=cmp_to_key(compare_for_building_block))
    if not allow_linear_combination:
        sorted_blocks = remove_linear_combination(sorted_blocks)
    if not allow_nested_blocks:
        sorted_blocks = remove_nested_blocks(sorted_blocks)
    building_blocks_in_orig_graph = restore_node_name_in_orig_graph(sorted_blocks, orig_graph)

    return building_blocks_in_orig_graph


def remove_nested_blocks(sorted_blocks: List[BuildingBlock]) -> List[BuildingBlock]:
    """
    Remove nested building blocks.

    :param: List of building blocks.
    :return: List of building blocks without nested blocks.
    """
    return [list(group_block)[-1] for _, group_block in groupby(sorted_blocks, lambda block: block.start_node.main_id)]


def get_group_of_dependent_blocks(blocks: List[BuildingBlock]) -> Dict[int, int]:
    """
    Building blocks can be categorized into groups. Blocks that follow each other in the graph
    (that is, they are connected by one edge) belong to the same group.

    :param: List of building blocks.
    :return: Dictionary where key is block index, value is group index.
    """
    groups = {}
    idx = 0
    groups = { idx: [] }
    for i in range(len(blocks) - 1):
        start_node_key_i1 = blocks[i + 1][0]
        end_node_key_i  = blocks[i][1]
        if start_node_key_i1 == end_node_key_i:
            groups[idx].append(i)
        else:
            groups[idx].append(i)
            idx +=1
            groups[idx] = []
    groups[idx].append(len(blocks) - 1)

    return groups


def get_building_blocks_info(bblocks: List[BuildingBlock], compressed_model: NNCFNetwork) -> List[BuildingBlockInfo]:
    """
    Returns additional information about building blocks.

    :param bblocks: List of building blocks.
    :param compressed_model: Target model.
    :return: List with additional info for each building blocks.
    """
    bblocks_info = []
    for block in bblocks:
        op_addresses = get_all_node_op_addresses_in_block(compressed_model, block)
        modules = get_all_modules_in_blocks(compressed_model, op_addresses)
        block_type = get_type_building_block(op_addresses)
        bblocks_info.append(BuildingBlockInfo(block, op_addresses, modules, block_type))
    return bblocks_info


def get_all_node_op_addresses_in_block(compressed_model: NNCFNetwork, block: BuildingBlock) -> Set[OperationAddress]:
    """
    Returns set of operation addresses of all layers included in the block.

    :param compressed_model: Target model.
    :param block: Building blocks.
    :return: Set of operation addresses for building block.
    """
    graph = compressed_model.get_original_graph()
    nx_graph = graph.get_nx_graph_copy()
    start_node, end_node = block
    start_node_key, end_node_key = None, None
    #pylint: disable=protected-access
    for node in nx_graph._node.values():
        if start_node == str(node['node_name']):
            start_node_key = node['key']
        if end_node == str(node['node_name']):
            end_node_key = node['key']
    simple_paths = nx.all_simple_paths(nx_graph, start_node_key, end_node_key)
    op_adresses = set()
    for node_keys_in_path in simple_paths:
        for node_key in node_keys_in_path:
            op_adresses.add(OperationAddress.from_str(nx_graph._node[node_key]['node_name']))
    start_op_address = OperationAddress.from_str(nx_graph._node[start_node_key]['node_name'])
    op_adresses.remove(start_op_address)
    return op_adresses


def get_all_modules_in_blocks(compressed_model: NNCFNetwork,
                              op_adresses_in_blocks: List[OperationAddress]) -> List[torch.nn.Module]:
    """
    Returns set of all modules included in the block.

    :param compressed_model: Target model.
    :param op_adresses_in_blocks: Set of operation addresses for building block.
    :return: List of module for building block.
    """
    modules = []
    for op_address in op_adresses_in_blocks:
        if op_address.operator_name in NNCF_MODULES_OP_NAMES:
            modules.append(compressed_model.get_module_by_scope(op_address.scope_in_model))
    return modules


def get_type_building_block(op_addresses_in_block: List[OperationAddress])-> BuildingBlockType:
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
        return BuildingBlockType.MSHA
    if count_fc == 2 and count_matmul == 0:
        return BuildingBlockType.FF
    return BuildingBlockType.Unknown
