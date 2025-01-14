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
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Set

import networkx as nx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.graph import NNCFNodeName
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns.manager import TargetDevice
from nncf.common.utils.backend import BackendType
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTRELUMetatype


class SearchGraphNode:
    """
    Class describing nodes used in SearchGraph.
    """

    def __init__(self, node_key: str, data: Dict):
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
        return self.data.get(NNCFNode.NODE_NAME_ATTR)

    @property
    def node_type(self) -> str:
        """
        Returns type of node.
        """
        return self.data.get(NNCFNode.NODE_TYPE_ATTR)

    @property
    def layer_name(self) -> str:
        """
        Returns the name of the layer to which the node corresponds.
        """
        return self.data.get(NNCFNode.LAYER_NAME_ATTR)

    @property
    def main_id(self) -> int:
        """
        Returns the id of node. In case if the node is merged returns id of the first node from merged node list.
        """
        if not self.is_merged:
            return self.data.get("id")
        return self.data.get(SearchGraph.MERGED_NODES_NODE_ATTR)[0].get("id")

    @property
    def bottom_id(self) -> int:
        """
        Returns the id of node. In case if the node is merged returns id of the last node from merged node list.
        """
        if not self.is_merged:
            return self.data.get("id")
        return self.data.get(SearchGraph.MERGED_NODES_NODE_ATTR)[-1].get("id")

    def __str__(self):
        return self.node_key

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, SearchGraphNode) and self.node_key == other.node_key


ShapeVsNodesMap = Dict[str, Set[SearchGraphNode]]


class SearchGraph:
    """
    A wrapper over the graph, which represents the DNN execution graph transformed
    by pattern matching, merging nodes and inserting auxiliary nodes.
    """

    ACTIVATION_OUTPUT_SHAPE_ATTR = "activation_output_shape"
    IS_MERGED_NODE_ATTR = "is_merged"
    IS_DUMMY_NODE_ATTR = "is_dummy"
    KEY_NODE_ATTR = "key"
    MERGED_NODES_NODE_ATTR = "merged_nodes"
    TYPE_NODE_ATTR = "type"
    DUMMY_POSTFIX = " dummy"

    def __init__(self, nx_merged_graph: nx.DiGraph):
        self._nx_graph = nx_merged_graph
        old_merged_graph_nodes = deepcopy(self._nx_graph._node)
        for node_key in old_merged_graph_nodes:
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

    def get_previous_nodes(self, node: SearchGraphNode) -> List[SearchGraphNode]:
        """
        Returns producer nodes of provided node.

        :param node: Consumer node.
        :return: List of producers nodes of provided node.
        """
        nx_node_keys = self._nx_graph.pred[node.node_key]
        return [self.get_node_by_key(node_key) for node_key in nx_node_keys]

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

    def _get_graph_for_visualization(self) -> nx.DiGraph:
        """
        :return: A user-friendly graph .dot file, making it easier to debug the network and setup
        ignored/target scopes.
        """
        out_graph = nx.DiGraph()
        for node in self.get_all_nodes():
            attrs_node = {}
            if node.is_merged:
                attrs_node["label"] = f"main: {node.main_id} bottom: {node.bottom_id} {node.node_key}"
            elif node.is_dummy:
                attrs_node["label"] = f"dummy {node.node_key}"
            else:
                attrs_node["label"] = f"id: {node.node_key}"
            out_graph.add_node(node.node_key, **attrs_node)

        for u, v in self._nx_graph.edges:
            edge = self._nx_graph.edges[u, v]
            style = "solid"
            out_graph.add_edge(u, v, label=edge[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR], style=style)

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")

        return out_graph

    def visualize_graph(self, path: str):
        out_graph = self._get_graph_for_visualization()
        write_dot_graph(out_graph, path)


def get_search_graph(original_graph: PTNNCFGraph, hw_fused_ops: bool) -> SearchGraph:
    """
    Returns a transformed representation of the network graph for blocks searching.
    """
    nx_merged_graph = get_merged_original_graph_with_pattern(original_graph.get_nx_graph_copy(), hw_fused_ops)
    sgraph = SearchGraph(nx_merged_graph)
    return sgraph


def get_merged_original_graph_with_pattern(orig_graph: nx.DiGraph, hw_fused_ops: bool) -> nx.DiGraph:
    """
    :param orig_graph: Original graph of model
    :param hw_fused_ops: indicates whether to merge operations by hw fusing pattern. Merged operation can't be part
    of different skipping block.
    :return: Graph with merged nodes by patterns
    """
    merged_graph = orig_graph
    if not hw_fused_ops:
        return merged_graph

    pattern_fusing_graph = PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH, device=TargetDevice.ANY)
    matches = find_subgraphs_matching_pattern(orig_graph, pattern_fusing_graph)
    nx.set_node_attributes(merged_graph, False, SearchGraph.IS_DUMMY_NODE_ATTR)
    nx.set_node_attributes(merged_graph, False, SearchGraph.IS_MERGED_NODE_ATTR)
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
            merged_node_key += str(attrs["id"]) + " " + attrs[SearchGraph.TYPE_NODE_ATTR] + "  "

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


def check_graph_has_no_hanging_edges_after_block_removal(
    graph: SearchGraph, first_skipped_node: SearchGraphNode, end_node: SearchGraphNode
) -> bool:
    """
    The subgraph is traversed starting with the first_skipped_node and ending with the end_node
    to determine that after deleting such a block there are no dangling edges in the graph.
    """
    #            A
    #           / \
    #          /   \
    #         /     \
    #    first_skipped_node  |   Edge A-C is dangling edges
    #        |       |
    #         \     /
    #          \   /
    #            C
    #            |
    #           ...
    #            |
    #         end_node
    start_node = first_skipped_node
    if not first_skipped_node.is_dummy:
        previous_nodes = graph.get_previous_nodes(first_skipped_node)
        num_inputs = len(previous_nodes)
        assert num_inputs == 1, f"building block should have a single input, but it has {num_inputs} inputs."
        start_node = previous_nodes[0]
    q = deque([start_node])
    addit_nodes = set()
    nodes = []
    potential_end_nodes = []
    while len(q) != 0:
        current_node = q.pop()
        if current_node.main_id != start_node.main_id:
            prev_nodes = graph.get_prev_nodes(current_node.node_key)
            if len(prev_nodes) > 1:
                for pn in prev_nodes:
                    if pn.bottom_id < start_node.bottom_id:
                        return False  # there is extra edge
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


def check_graph_has_no_duplicate_edges_after_block_removal(
    sgraph: SearchGraph, first_skipped_node: SearchGraphNode, end_node: SearchGraphNode
) -> bool:
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

    if first_skipped_node.is_dummy:
        next_end_node = sgraph.get_next_nodes(end_node.node_key)
        if len(next_end_node) != 0:
            attr = sgraph._nx_graph.get_edge_data(first_skipped_node.node_key, next_end_node[0].node_key)
        else:
            attr = None
    else:
        previous_node = sgraph.get_prev_nodes(first_skipped_node.node_key)
        next_end_node = sgraph.get_next_nodes(end_node.node_key)
        if len(previous_node) != 0 and len(next_end_node) != 0:
            attr = sgraph._nx_graph.get_edge_data(previous_node[0].node_key, next_end_node[0].node_key)
        else:
            attr = None
    return attr is None


def check_graph_has_no_act_layer_duplication_after_block_removal(
    sgraph: SearchGraph, first_skipped_node: SearchGraphNode, end_node: SearchGraphNode
) -> bool:
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

    previous_nodes = sgraph.get_prev_nodes(first_skipped_node.node_key)
    next_end_node = sgraph.get_next_nodes(end_node.node_key)
    if len(next_end_node) == 0 or len(previous_nodes) == 0:
        return True
    if previous_nodes[0].is_dummy:
        previous_nodes = sgraph.get_prev_nodes(previous_nodes[0].node_key)

    if (
        previous_nodes[0].node_type[-1] in PTRELUMetatype.get_all_aliases()
        and next_end_node[0].node_type[0] in PTRELUMetatype.get_all_aliases()
    ):
        return False
    return True


def get_num_ops_in_block(first_skipped_node: SearchGraphNode, end_node: SearchGraphNode) -> int:
    """
    Calculates number of operations in the block by using indexes. Indexes should be in execution order.
    The block is defined by first and last skipped node.
    """
    num_ops = end_node.bottom_id - first_skipped_node.main_id
    if first_skipped_node.is_dummy:
        num_ops -= 1
    return num_ops
