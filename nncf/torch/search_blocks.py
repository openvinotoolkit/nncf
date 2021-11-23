from copy import deepcopy
from functools import cmp_to_key
from itertools import combinations

import networkx as nx
from nncf.common.graph.graph import NNCFGraph, LayerName
from typing import List, Dict, Type
# from graphviz import Digraph
from nncf.torch.hardware.fused_patterns import PT_HW_FUSED_PATTERNS
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.torch.graph.operator_metatypes import DropoutMetatype
from nncf.common.graph.definitions import MODEL_OUTPUT_OP_NAME
from nncf.torch.dynamic_graph.operation_address import OperationAddress
from nncf.torch.layers import NNCF_MODULES_MAP, NNCF_MODULES_OP_NAMES


class SearchGraphNode:
    """
    Class describing nodes used in SGraph.
    """

    def __init__(self,
                 node_key: str,
                 data: dict = None):
        self.node_key = node_key
        self.data = data if data else {}

    @property
    def is_merged(self):
        return self.data.get(SearchGraph.IS_MERGED_NODE_ATTR)

    @property
    def is_dummy(self):
        return self.data.get(SearchGraph.IS_DUMMY_NODE_ATTR)

    @property
    def node_name(self) -> str:
        return self.data.get(NNCFGraph.NODE_NAME_ATTR)

    @property
    def metatype(self) -> Type[OperatorMetatype]:
        return self.data.get(NNCFGraph.METATYPE_ATTR)

    @property
    def node_type(self) -> str:
        return self.data.get(NNCFGraph.NODE_TYPE_ATTR)

    @property
    def layer_name(self) -> LayerName:
        return self.data.get(NNCFGraph.LAYER_NAME_ATTR)

    @property
    def main_id(self) -> int:
        # return the first id node
        if not self.is_merged:
            return self.data.get('id')
        else:
            return self.data.get(SearchGraph.MERGED_NODES_NODE_ATTR)[0].get('id')

    @property
    def bottom_id(self) -> int:
        if not self.is_merged:
            return self.data.get('id')
        else:
            return self.data.get(SearchGraph.MERGED_NODES_NODE_ATTR)[-1].get('id')

    def __str__(self):
        return self.node_key

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, SearchGraphNode) and self.node_key == other.node_key


class SearchGraph():

    IS_MERGED_NODE_ATTR = 'is_merged'
    IS_DUMMY_NODE_ATTR = 'is_dummy'
    ACTIVATION_INPUT_SHAPE_ATTR = 'activation_input_shape'
    ACTIVATION_OUTPUT_SHAPE_ATTR = 'activation_output_shape'
    KEY_NODE_ATTR = 'key'
    TYPE_NODE_ATTR = 'type'
    MERGED_NODES_NODE_ATTR = 'merged_nodes'


    def __init__(self, nx_merged_graph):
        self._nx_graph = nx_merged_graph

    def _nx_node_to_sgraph_node(self, nx_node_key, nx_node_attrs):
        return SearchGraphNode(nx_node_key, nx_node_attrs)

    def get_node_by_key(self, node_key):
        return SearchGraphNode(node_key, self._nx_graph.nodes[node_key])

    def get_all_nodes(self):
        all_nodes = []
        for node_key, node_attrs in self._nx_graph.nodes.items():
            all_nodes.append(SearchGraphNode(node_key, node_attrs))
        return all_nodes

    def get_next_nodes(self, node_key):
        next_node_keys = self._nx_graph.succ[node_key]
        return [self.get_node_by_key(node_key) for node_key in next_node_keys]

    def get_prev_nodes(self, node_key):
        prev_node_keys = self._nx_graph.pred[node_key]
        return [self.get_node_by_key(node_key) for node_key in prev_node_keys]

    def get_prev_edges(self, node_key):
        return self._nx_graph.pred[node_key]

    def get_next_edges(self, node_key):
        return self._nx_graph.succ[node_key]

    def set_node_attr(self, node_key, name_attr, value_attr):
        self._nx_graph.nodes[node_key][name_attr] = value_attr

IGNORED_NAME_OPERATORS = [DropoutMetatype.name, MODEL_OUTPUT_OP_NAME]

def prepare_search_graph(original_graph: NNCFGraph) -> SearchGraph:
    nx_merged_graph = get_orig_graph_with_orig_pattern(original_graph._nx_graph)
    sgraph = SearchGraph(nx_merged_graph)

    return sgraph

def add_node(d, shape, node):
    if str(shape) in d:
        d[str(shape)].append(node)
    else:
        d[str(shape)] = [node]

def insert_dummy_node(nx_graph, node_key):
    next_nodes = deepcopy(nx_graph._succ[node_key])
    dummy_node_attr = {SearchGraph.IS_DUMMY_NODE_ATTR: True, 'has_dummy': False}
    node_key_attr = deepcopy(nx_graph._node[node_key])
    dummy_node_key = node_key + ' dummy'
    node_key_attr.update(dummy_node_attr)
    nx_graph.add_node(dummy_node_key, **node_key_attr)

    edge_attrs = None
    for next_node_key in next_nodes:
        edge_attrs = nx_graph.get_edge_data(node_key, next_node_key)
        nx_graph.remove_edge(node_key, next_node_key)
        nx_graph.add_edge(dummy_node_key, next_node_key, **edge_attrs)
    nx_graph.add_edge(node_key, dummy_node_key, **edge_attrs)


def get_orig_graph_with_orig_pattern(orig_graph):
    from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern

    # pylint: disable=protected-access
    pattern_fusing_graph = PT_HW_FUSED_PATTERNS.get_full_pattern_graph()
    matches = find_subgraphs_matching_pattern(orig_graph, pattern_fusing_graph)
    merged_graph = deepcopy(orig_graph)
    nx.set_node_attributes(merged_graph, False, SearchGraph.IS_DUMMY_NODE_ATTR)
    nx.set_node_attributes(merged_graph, False, SearchGraph.IS_MERGED_NODE_ATTR)
    nx.set_node_attributes(merged_graph, None, SearchGraph.ACTIVATION_INPUT_SHAPE_ATTR)
    nx.set_node_attributes(merged_graph, None, SearchGraph.ACTIVATION_OUTPUT_SHAPE_ATTR)
    nx.set_node_attributes(merged_graph, False, 'has_dummy')
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
            'has_dummy': False
        }
        merged_graph.add_node(merged_node_key, **merged_node_attrs)
        for in_edge_key, in_edge_attrs in in_edge_copies_dict.items():
            merged_graph.add_edge(in_edge_key[0], merged_node_key, **in_edge_attrs)
        for out_edge_key, out_edge_attrs in out_edge_copies_dict.items():
            merged_graph.add_edge(merged_node_key, out_edge_key[1], **out_edge_attrs)

    old_merged_graph_nodes = deepcopy(merged_graph._node)
    for node_key in old_merged_graph_nodes:
        next_nodes = merged_graph._succ[node_key]
        if len(list(next_nodes)) > 1:
            merged_graph.nodes[node_key]['has_dummy'] = True
            insert_dummy_node(merged_graph, node_key)

    return merged_graph


def add_node_to_aux_struct(node_key, shape, shape_map):
    str_shape = str(shape)
    if str_shape in shape_map:
        shape_map[str_shape].add(node_key)
    else:
        shape_map[str_shape] = set([node_key])

def comparator(u, v):
    return u.main_id - v.main_id

def traverse_subgraph(graph: SearchGraph, start_node: SearchGraphNode, end_node: SearchGraphNode):
    from collections import deque
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
    else:
        for node in addit_nodes:
            if node not in nodes:
                return False
        nodes.append(end_node)
        return True

def traverse_subgraph_last(graph: SearchGraph, start_node: SearchGraphNode, end_node: SearchGraphNode):
    from collections import deque
    q = deque([start_node])
    addit_nodes = set()
    nodes = []
    current_node = start_node
    potential_end_nodes = []
    while len(q) != 0:
        current_node = q.pop()
        if current_node.node_key != start_node.node_key:
            prev_nodes = graph.get_prev_nodes(current_node.node_key)
            if len(prev_nodes) > 1:
                for pn in prev_nodes:
                    addit_nodes.add(pn)
        if current_node.node_key == end_node.node_key:
            continue
        if current_node.main_id > end_node.main_id:
            return False
        nodes.append(current_node)
        next_nodes = graph.get_next_nodes(current_node.node_key)
        if len(next_nodes) == 0:
            potential_end_nodes.append(current_node)
        for next_node in next_nodes:
            q.appendleft(next_node)
    if len(q) > 0 or len(potential_end_nodes) > 0:
        return False
    else:
        for node in addit_nodes:
            if node not in nodes:
                return False
        nodes.append(end_node)
        return True


def rule__not_lead_to_double_edge(sgraph: SearchGraph, start_node: SearchGraphNode, end_node: SearchGraphNode):
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
    return True if attr is None else False

def rule__not_lead_to_duplication_of_activation(sgraph: SearchGraph, start_node: SearchGraphNode, end_node: SearchGraphNode):
    pred_start_node = sgraph.get_prev_nodes(start_node.node_key)
    next_end_node = sgraph.get_next_nodes(end_node.node_key)
    if len(next_end_node) == 0 or len(pred_start_node) == 0:
        return True
    if pred_start_node[0].is_dummy:
        pred_start_node = sgraph.get_prev_nodes(pred_start_node[0].node_key)

    if 'relu' == pred_start_node[0].node_type[-1] and 'relu' == next_end_node[0].node_type[0]:
        return False
    return True


def comparator_for_block(block1, block2):
    if block1[1].bottom_id != block2[1].bottom_id:
        return block1[1].bottom_id - block2[1].bottom_id
    else:
        return block2[0].main_id - block1[0].main_id

def blocks_combination_is_block(block, combination):

    if block[0].main_id != combination[0][0].main_id:
        return False
    if block[1].bottom_id != combination[-1][1].bottom_id:
        return False
    i = 0
    while i < len(combination) - 1:
        end_i_node = combination[i][1]
        start_nexti_node = combination[i + 1][0]
        if end_i_node.node_key in start_nexti_node.node_key:
            i += 1
            continue
        else:
            return False
    return True

def search_lin_combination(block, blocks):
    max_num = len(blocks)
    all_combinations = []
    for i in range(max_num, 1, -1):
        all_combinations = list(combinations(blocks, i))
        for combo in all_combinations:
            if blocks_combination_is_block(block, combo):
                return True
    return False

def remove_lin_combination(sorted_building_blocks):
    good_blocks = []
    start_to_idx = {}
    last_node = None
    for i, block in enumerate(sorted_building_blocks):
        start_node, end_node = block
        #if end_node.main_id - start_node.main_id > 23:
        #    continue
        if last_node == end_node:
            if start_node.main_id in start_to_idx:
                if not search_lin_combination(block, good_blocks[start_to_idx[start_node.main_id]:]):
                    good_blocks.append(block)
                    last_node = end_node
        else:
            good_blocks.append(block)
            last_node = end_node
            if start_node.main_id not in start_to_idx:
                start_to_idx[start_node.main_id] = len(good_blocks) - 1

    return good_blocks

def match_blocks_to_full_node_names(building_blocks, orig_graph):

    building_block_in_orig_format = []
    for block in building_blocks:
        start_node, end_node = block
        #if start_node.is_merged:
        id_st = start_node.bottom_id # dummy node
        id_end = end_node.bottom_id
        block_in_orig_format = [orig_graph.get_node_key_by_id(id_st).split(' ')[-1],
                                orig_graph.get_node_key_by_id(id_end).split(' ')[-1]]
        building_block_in_orig_format.append(block_in_orig_format)
    return building_block_in_orig_format

def get_building_blocks(compressed_model, max_block_size=50, allow_nested_blocks=True):

    orig_graph = compressed_model.get_original_graph() # PTNNCFGraph
    sgraph = prepare_search_graph(orig_graph)

    act_input_shape = {} # key - str(shape), value - set of node_key
    act_output_shape = {} # key - str(shape), value - set of node_keysss

    blocks = []
    for node in sgraph.get_all_nodes():
        next_edges = sgraph.get_next_edges(node.node_key)
        prev_edges = sgraph.get_prev_edges(node.node_key)

        for _, edge_attr in next_edges.items():
            sgraph.set_node_attr(node.node_key, 'activation_output_shape', edge_attr['activation_shape'])
            break
        add_node_to_aux_struct(node, edge_attr['activation_shape'], act_output_shape)
        for _, edge_attr in prev_edges.items():
            sgraph.set_node_attr(node.node_key, 'activation_input_shape', edge_attr['activation_shape'])
            break
        if node.node_key == '40 conv2d  41 relu_   dummy':
            st = 1
        #if not node.data['has_dummy'] or node.is_dummy:
        add_node_to_aux_struct(node, edge_attr['activation_shape'], act_input_shape)
        #else:
        #    st = 1

    for shape, start_nodes in act_input_shape.items():
        for start_node in start_nodes:
            if start_node.node_type == IGNORED_NAME_OPERATORS:
                continue
            pred_start_node = sgraph.get_prev_nodes(start_node.node_key)
            if len(pred_start_node) != 1:
                # in this node several diff. edges enter
                continue
            for end_node in act_output_shape[shape]:
                if end_node.main_id - start_node.main_id > max_block_size:
                    continue
                if end_node.node_type in IGNORED_NAME_OPERATORS:
                    continue
                if end_node.node_key == start_node.node_key:
                    continue
                if end_node.is_dummy:
                    continue
                if end_node.node_key in start_node.node_key:
                    continue
                if end_node.main_id < start_node.main_id:
                    continue
                is_one_edge = (end_node.main_id - start_node.bottom_id) == 1 

                # CHECK RULES
                status_0 = rule__not_lead_to_double_edge(sgraph, start_node, end_node)
                status_1 = rule__not_lead_to_duplication_of_activation(sgraph, start_node, end_node)
                if status_0 and status_1 and not is_one_edge:
                    status_2 = traverse_subgraph(sgraph, start_node, end_node)
                    if status_2:
                        blocks.append((start_node, end_node))

    sorted_blocks = sorted(blocks, key=cmp_to_key(comparator_for_block))
    sorted_blocks = remove_lin_combination(sorted_blocks)
    if not allow_nested_blocks:
        sorted_blocks = remove_nested_blocks(sorted_blocks)
    blocks_in_orig_format = match_blocks_to_full_node_names(sorted_blocks, orig_graph)

    get_all_modules_per_blocks(compressed_model, blocks_in_orig_format)
    return blocks_in_orig_format


from itertools import groupby
def remove_nested_blocks(sorted_blocks):
    return [list(group_block)[-1] for _, group_block in groupby(sorted_blocks, lambda block: block[0].main_id)]

def get_group_of_dependent_blocks(blocks):
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


def get_all_node_op_addresses_in_block(compressed_model, blocks):
    graph = compressed_model.get_original_graph()
    all_nodes_per_skipped_block_idxs = {}
    for idx, block in enumerate(blocks):
        start_node, end_node = block
        start_node_key, end_node_key = None, None
        for node in graph._nx_graph._node.values():
            if start_node == str(node['node_name']):
                start_node_key = node['key']
            if end_node == str(node['node_name']):
                end_node_key = node['key']
        simple_paths = nx.all_simple_paths(graph._nx_graph, start_node_key, end_node_key)
        all_nodes_in_block = set()
        for node_keys_in_path in simple_paths:
            for node_key in node_keys_in_path:
                all_nodes_in_block.add(str(graph._nx_graph._node[node_key]['node_name']))
        start_op_address = str(graph._nx_graph._node[start_node_key]['node_name'])
        all_nodes_in_block.remove(start_op_address)
        all_nodes_per_skipped_block_idxs[idx] = list(all_nodes_in_block)
    return all_nodes_per_skipped_block_idxs


def get_all_modules_per_blocks(compressed_model, blocks):
    modules_per_idxs = {}
    all_node_op_addr_in_blocks = get_all_node_op_addresses_in_block(compressed_model, blocks)
    for idx, nodes_per_block in all_node_op_addr_in_blocks.items():
        modules_per_idxs[idx] = []
        for str_op_addr in nodes_per_block:
            op_address = OperationAddress.from_str(str_op_addr)
            if op_address.operator_name in NNCF_MODULES_OP_NAMES:
                modules_per_idxs[idx].append(compressed_model.get_module_by_scope(op_address.scope_in_model))
    return modules_per_idxs

# def vis(graph, blocks):
#     # correct work only for visualization blocks without common vertices
#     dot = Digraph()
#     for id, block_path in enumerate(blocks):
#         with dot.subgraph(name='cluster_{}'.format(id)) as c:
#             c.attr(color='blue')
#             c.node_attr['style'] = 'filled'
#             for node in block_path:
#                 c.node(node)
#             c.attr(label='block #{}'.format(id))
#
#     for node in graph.nodes:
#         dot.node(graph.nodes[node]['key'])
#         for child in graph.successors(node):
#             dot.edge(node, child)
#     return dot
