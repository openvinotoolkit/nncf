"""
 Copyright (c) 2019-2020 Intel Corporation
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

from typing import List
from typing import Optional
from typing import Callable
from typing import Tuple
from typing import Any

import networkx as nx


class NNCFNode:
    """
    Class describing nodes used in NNCFGraph
    """
    def __init__(self,
                 node_id: int,
                 data: dict = None):
        self.node_id = node_id
        self.data = data if data else {}

    @property
    def node_name(self):
        return self.data.get(NNCFGraph.KEY_NODE_ATTR)

    @property
    def node_type(self):
        return self.data.get(NNCFGraph.NODE_TYPE_ATTR)

    @property
    def module_attributes(self):
        return self.data.get(NNCFGraph.MODULE_ATTRIBUTES)

    def __str__(self):
        return ' '.join([self.node_id, self.node_name, self.node_type])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, NNCFNode) \
               and self.node_id == other.node_id \
               and self.data == other.data \
               and self.node_type == other.node_type \
               and self.module_attributes == other.module_attributes


class NNCFGraph:
    """
    Wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN
    providing some useful methods for graph traverse
    """
    ID_NODE_ATTR = 'id'
    KEY_NODE_ATTR = 'key'
    NODE_TYPE_ATTR = 'type'
    MODULE_ATTRIBUTES = 'module_attributes'

    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict = dict()

    def get_node_by_id(self, node_id) -> NNCFNode:
        return self._nx_node_to_nncf_node(self._nx_graph.nodes[self.get_node_key_by_id(node_id)])

    def get_input_nodes(self) -> List[NNCFNode]:
        inputs = []
        for nx_node_key, deg in self._nx_graph.in_degree():
            if deg == 0:
                inputs.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[nx_node_key]))
        return inputs

    def get_graph_outputs(self) -> List[NNCFNode]:
        outputs = []
        for nx_node_key, deg in self._nx_graph.out_degree():
            if deg == 0:
                outputs.append(self._nx_node_to_nncf_node(self._nx_graph.nodes[nx_node_key]))
        return outputs

    def get_all_node_idxs(self):
        return self._node_id_to_key_dict.keys()

    def get_nodes_by_types(self, type_list: List[str]) -> List[NNCFNode]:
        all_nodes_of_type = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            if nncf_node.node_type in type_list:
                all_nodes_of_type.append(nncf_node)
        return all_nodes_of_type

    def get_all_node_keys(self):
        return self._node_id_to_key_dict.copy().values()

    def get_all_nodes(self) -> List[NNCFNode]:
        all_nodes = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            all_nodes.append(nncf_node)
        return all_nodes

    @staticmethod
    def _nx_node_to_nncf_node(nx_node: dict) -> NNCFNode:
        return NNCFNode(nx_node[NNCFGraph.ID_NODE_ATTR], nx_node)

    def get_node_key_by_id(self, node_id):
        return self._node_id_to_key_dict[node_id]

    def get_nx_node_by_key(self, key: str):
        return self._nx_graph.nodes[key]

    def get_next_nodes(self, node: NNCFNode) -> Optional[List[NNCFNode]]:
        nx_node_keys = self._nx_graph.succ[self._node_id_to_key_dict[node.node_id]]
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def get_previous_nodes(self, node: NNCFNode) -> Optional[List[NNCFNode]]:
        nx_node_keys = self._nx_graph.pred[self._node_id_to_key_dict[node.node_id]]
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def traverse_graph(self,
                       curr_node: NNCFNode,
                       traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
                       traverse_forward: bool = True):
        output = []
        return self._traverse_graph_recursive_helper(curr_node, traverse_function, output, traverse_forward)

    def _traverse_graph_recursive_helper(self, curr_node: NNCFNode,
                                         traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
                                         output: List[Any], traverse_forward: bool):
        is_finished, output = traverse_function(curr_node, output)
        get_nodes_fn = self.get_next_nodes if traverse_forward else self.get_previous_nodes
        if not is_finished:
            for node in get_nodes_fn(curr_node):
                self._traverse_graph_recursive_helper(node, traverse_function, output, traverse_forward)
        return output

    def add_node(self, label, **attrs):
        node_id = len(self._node_id_to_key_dict)
        self._node_id_to_key_dict[node_id] = label
        attrs[NNCFGraph.KEY_NODE_ATTR] = label
        attrs[NNCFGraph.ID_NODE_ATTR] = node_id
        self._nx_graph.add_node(label, **attrs)

    def add_edge(self, u_of_edge, v_of_edge, **attrs):
        self._nx_graph.add_edge(u_of_edge, v_of_edge, **attrs)

    def topological_sort(self):
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[node_name])
                for node_name in nx.topological_sort(self._nx_graph)]

    def dump_graph(self, path):
        nx.drawing.nx_pydot.write_dot(self._nx_graph, path)
