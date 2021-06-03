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

import networkx as nx

from collections import OrderedDict
from typing import Any, Callable, Dict, KeysView, List, Tuple, Type, ValuesView

from nncf.common.graph.module_attributes import BaseModuleAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import get_input_metatypes
from nncf.common.graph.operator_metatypes import get_output_metatypes

MODEL_INPUT_OP_NAME = "nncf_model_input"
MODEL_OUTPUT_OP_NAME = "nncf_model_output"


class NNCFNode:
    """
    Class describing nodes used in NNCFGraph.
    """

    def __init__(self,
                 node_id: int,
                 data: dict = None):
        self.node_id = node_id
        self.data = data if data else {}

    @property
    def node_name(self) -> str:
        return self.data.get(NNCFGraph.KEY_NODE_ATTR)

    @property
    def node_type(self) -> str:
        return self.data.get(NNCFGraph.NODE_TYPE_ATTR)

    @property
    def metatype(self) -> Type[OperatorMetatype]:
        return self.data.get(NNCFGraph.METATYPE_ATTR)

    @property
    def module_attributes(self) -> BaseModuleAttributes:
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


class NNCFGraphNodeType:
    INPUT_NODE = MODEL_INPUT_OP_NAME
    OUTPUT_NODE = MODEL_OUTPUT_OP_NAME


class NNCFGraph:
    """
    Wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN
    providing some useful methods for graph traversal.
    """

    ID_NODE_ATTR = 'id'
    KEY_NODE_ATTR = 'key'
    NODE_TYPE_ATTR = 'type'
    METATYPE_ATTR = 'metatype'
    MODULE_ATTRIBUTES = 'module_attributes'
    ACTIVATION_SHAPE_EDGE_ATTR = 'activation_shape'
    IN_PORT_NAME_EDGE_ATTR = 'in_port'

    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict = dict()
        self._input_nncf_nodes = {}  # type: Dict[int, NNCFNode]
        self._output_nncf_nodes = {}  # type: Dict[int, NNCFNode]

    @property
    def nx_graph(self) -> nx.DiGraph:
        """
        Returns the internal representation of the graph as a DiGraph instance.

        :return: The DiGraph instance.
        """
        return self._nx_graph

    def get_node_by_id(self, node_id: int) -> NNCFNode:
        """
        :param node_id: Id of the node.
        :return: Node in a graph with such id.
        """
        return self.get_node_by_key(self.get_node_key_by_id(node_id))

    def get_node_by_key(self, key: str):
        """
        :param key: key (node_name) of the node.
        :return: NNCFNode in a graph with such key.
        """
        return self._nx_node_to_nncf_node(self._nx_graph.nodes[key])

    def get_input_nodes(self) -> List[NNCFNode]:
        return list(self._input_nncf_nodes.values())

    def get_output_nodes(self) -> List[NNCFNode]:
        return list(self._output_nncf_nodes.values())

    def get_nodes_by_types(self, type_list: List[str]) -> List[NNCFNode]:
        """
        :param type_list: List of types to look for.
        :return: List of nodes with provided types.
        """
        all_nodes_of_type = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            if nncf_node.node_type in type_list:
                all_nodes_of_type.append(nncf_node)
        return all_nodes_of_type

    def get_nodes_by_metatypes(self, metatype_list: List[Type[OperatorMetatype]]) -> List[NNCFNode]:
        """
        Return a list of nodes with provided metatypes.

        :param metatype_list: List of types to look for.
        :return: List of nodes with provided metatypes.
        """
        all_nodes_of_type = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            if nncf_node.metatype in metatype_list:
                all_nodes_of_type.append(nncf_node)
        return all_nodes_of_type


    def get_all_node_ids(self) -> KeysView[int]:
        """
        Returns all graph nodes' node_ids.
        """
        return self._node_id_to_key_dict.keys()

    def get_all_node_keys(self) -> ValuesView[str]:
        """
        Returns all graph nodes' keys i.e. node_names.
        """
        return self._node_id_to_key_dict.copy().values()

    def get_all_nodes(self) -> List[NNCFNode]:
        """
        Returns list of all graph nodes.
        """
        all_nodes = []
        for node_key in self.get_all_node_keys():
            nx_node = self._nx_graph.nodes[node_key]
            nncf_node = self._nx_node_to_nncf_node(nx_node)
            all_nodes.append(nncf_node)
        return all_nodes

    @staticmethod
    def _nx_node_to_nncf_node(nx_node: dict) -> NNCFNode:
        return NNCFNode(nx_node[NNCFGraph.ID_NODE_ATTR], nx_node)

    def get_node_key_by_id(self, node_id: id) -> str:
        """
        Returns node key (node_name) by provided id.

        :param node_id: Id of the node.
        :return: Key of the node with provided id.
        """
        return self._node_id_to_key_dict[node_id]

    def get_next_nodes(self, node: NNCFNode) -> List[NNCFNode]:
        """
        Returns consumer nodes of provided node.

        :param node: Producer node.
        :return: List of consumer nodes of provided node.
        """
        nx_node_keys = self._nx_graph.succ[self._node_id_to_key_dict[node.node_id]]
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def get_previous_nodes(self, node: NNCFNode) -> List[NNCFNode]:
        """
        Returns producer nodes of provided node.

        :param node: Consumer node.
        :return: List of producers nodes of provided node.
        """

        nx_node_keys = self._nx_graph.pred[self._node_id_to_key_dict[node.node_id]]
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[key]) for key in nx_node_keys]

    def get_input_edges(self, node: NNCFNode) -> Dict[Tuple[str, str], dict]:
        """
        Returns edges of input tensors with description sorted by 'in_port'.

        :param node: Consumer node.
        :return: Dictionary of input edges for node sorted by in_port.
        """
        nx_node_key = self._node_id_to_key_dict[node.node_id]
        input_edges = sorted(list(self._nx_graph.in_edges(nx_node_key)),
                             key=lambda edge: self._nx_graph.edges[edge][NNCFGraph.IN_PORT_NAME_EDGE_ATTR])

        return OrderedDict((edge, self._nx_graph.edges[edge]) for edge in input_edges)

    def get_output_edges(self, node: NNCFNode) -> Dict[Tuple[str, str], dict]:
        """
        Returns edges of output tensors with description. Unordered.

        :param node: Producer node.
        :return: Dictionary of output edges for the node.
        """
        nx_node_key = self._node_id_to_key_dict[node.node_id]
        return {edge: self._nx_graph.edges[edge] for edge in self._nx_graph.out_edges(nx_node_key)}

    def traverse_graph(self,
                       curr_node: NNCFNode,
                       traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
                       traverse_forward: bool = True):
        """
        Traverses graph up or down starting form `curr_node` node.

        :param curr_node: Node from which traversal is started.
        :param traverse_function: Function describing condition of traversal continuation/termination.
        :param traverse_forward: Flag specifying direction of traversal.
        :return:
        """
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

    def add_node(self, label: str, **attrs):
        """
        Adds node with `label` key (node_name) and `attr` attributes to the graph.

        :param label: Key (node_name) of the node.
        :param attrs: Attributes of the node.
        """
        node_id = len(self._node_id_to_key_dict)
        attrs[NNCFGraph.ID_NODE_ATTR] = node_id
        self._node_id_to_key_dict[node_id] = label
        attrs[NNCFGraph.KEY_NODE_ATTR] = label
        self._nx_graph.add_node(label, **attrs)

        nncf_node = self.get_node_by_id(node_id)
        if nncf_node.metatype in get_input_metatypes():
            self._input_nncf_nodes[node_id] = nncf_node

        if nncf_node.metatype in get_output_metatypes():
            self._output_nncf_nodes[node_id] = nncf_node

    def add_edge(self, u_of_edge: str, v_of_edge: str, **attrs):
        """
        Adds edge with attrs attributes between u and v.

        :param u_of_edge: Producer node key (node_name).
        :param v_of_edge: Consumer node key (node_name).
        :param attrs: Attributes of the edge.
        """
        self._nx_graph.add_edge(u_of_edge, v_of_edge, **attrs)

    def topological_sort(self) -> List[NNCFNode]:
        """
        Returns nodes in topologically sorted order, additionally sorted in ascending node ID order.
        """
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[node_name])
                for node_name in
                nx.lexicographical_topological_sort(self._nx_graph,
                                                    key=lambda x: self._nx_graph.nodes[x][NNCFGraph.ID_NODE_ATTR])]

    def dump_graph(self, path: str):
        """
        Writes graph in .dot format.

        :param path: Path to save.
        """
        nx.drawing.nx_pydot.write_dot(self._nx_graph, path)
