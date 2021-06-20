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
import os
from collections import OrderedDict
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, KeysView, List, Tuple, Type, ValuesView

import networkx as nx
import networkx.algorithms.isomorphism as iso
from networkx.drawing.nx_agraph import to_agraph

from nncf.common.graph.graph_matching import Expression
from nncf.common.graph.graph_matching import NodeExpression
from nncf.common.graph.graph_matching import find_subgraphs_matching_expression
from nncf.common.graph.graph_matching import get_edge_boundaries
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.operator_metatypes import get_input_metatypes
from nncf.common.graph.operator_metatypes import get_output_metatypes
from nncf.common.utils.logger import logger as nncf_logger

MODEL_INPUT_OP_NAME = 'nncf_model_input'
MODEL_OUTPUT_OP_NAME = 'nncf_model_output'

NNCFNodeName = str
LayerName = str


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
    def node_name(self) -> NNCFNodeName:
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
    def layer_attributes(self) -> BaseLayerAttributes:
        return self.data.get(NNCFGraph.LAYER_ATTRIBUTES)

    @property
    def ignored_algorithms(self) -> List[str]:
        return self.data.get(NNCFGraph.IGNORED_ALGOS_ATTR, [])

    def is_in_iteration_scope(self) -> bool:
        return self.data.get(NNCFGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR, False)

    def is_integer_input(self) -> bool:
        return self.data.get(NNCFGraph.IS_INTEGER_INPUT_NODE_ATTR, False)

    def is_shared(self) -> bool:
        return self.data.get(NNCFGraph.IS_SHARED_ATTR, False)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return ' '.join([str(self.node_id), self.node_name, self.node_type])

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return isinstance(other, NNCFNode) \
               and self.node_id == other.node_id \
               and self.data == other.data \
               and self.node_type == other.node_type \
               and self.layer_attributes == other.layer_attributes


class NNCFGraphNodeType:
    INPUT_NODE = MODEL_INPUT_OP_NAME
    OUTPUT_NODE = MODEL_OUTPUT_OP_NAME


class NNCFGraphEdge:
    """
    A structure describing an edge in NNCFGraph. Since nodes of the NNCFGraph are operations
    on (activation) tensors, an edge in NNCFGraph is a representation of an activation tensor produced or
    consumed by an operation.
    """
    def __init__(self, from_node: NNCFNode, to_node: NNCFNode, tensor_shape: List[int]):
        """
        :param from_node: An NNCFNode that sources the directed edge.
        :param to_node: An NNCFNode that sinks the directed edge.
        :param tensor_shape: The shape of the activation tensor the edge represents.
        """
        self.from_node = from_node
        self.to_node = to_node
        self.tensor_shape = tensor_shape

    def __str__(self):
        return str(self.from_node) + ' -> ' + str(self.tensor_shape) + ' -> ' + str(self.to_node)

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return self.from_node == other.from_node and self.to_node == other.to_node \
               and self.tensor_shape == other.tensor_shape


class NNCFGraphPatternIO:
    """
    Describes the inputs and outputs of a subgraph in NNCFGraph.
    """
    def __init__(self, input_edges: List[NNCFGraphEdge], output_edges: List[NNCFGraphEdge]):
        self.input_edges = input_edges
        self.output_edges = output_edges


class NNCFNodeExpression(NodeExpression):
    """
    A variation of NodeExpression that has a specific matcher function to work with nxgraph node dicts that
    underlie the NNCFGraph.
    """
    def __init__(self, node_type: str = None, filter_fn=None):
        node_type_fn = lambda x: x[NNCFGraph.NODE_TYPE_ATTR]
        super().__init__(node_type, filter_fn, node_type_fn=node_type_fn)


#pylint:disable=too-many-public-methods
class NNCFGraph:
    """
    Wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN
    providing some useful methods for graph traversal.
    """

    ID_NODE_ATTR = 'id'
    KEY_NODE_ATTR = 'key'
    NODE_NAME_ATTR = 'node_name'
    NODE_TYPE_ATTR = 'type'
    METATYPE_ATTR = 'metatype'
    LAYER_NAME_ATTR = 'layer_name'
    LAYER_ATTRIBUTES = 'layer_attributes'
    ACTIVATION_SHAPE_EDGE_ATTR = 'activation_shape'
    IN_PORT_NAME_EDGE_ATTR = 'in_port'
    IGNORED_ALGOS_ATTR = 'ignored_algos'
    IS_IN_ITERATION_SCOPE_NODE_ATTR = 'is_in_iteration_scope'
    IS_INTEGER_INPUT_NODE_ATTR = 'is_integer_input'
    DTYPE_EDGE_ATTR = 'dtype'
    IS_SHARED_ATTR = 'is_shared'

    def __init__(self):
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict = dict()
        self._input_nncf_nodes = {}  # type: Dict[int, NNCFNode]
        self._output_nncf_nodes = {}  # type: Dict[int, NNCFNode]

        self._node_ids_vs_layer_names = {}  # type: Dict[int, LayerName]
        self._layer_name_vs_shared_nodes = defaultdict(list)  # type: Dict[LayerName, List[NNCFNode]]

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
        """
        :return: List of input nodes of the graph.
        """
        return list(self._input_nncf_nodes.values())

    def get_output_nodes(self) -> List[NNCFNode]:
        """
        :return: List of output nodes of the graph.
        """
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

    def add_nncf_node(self, node_name: str,
                      node_type: str,
                      node_metatype: Type[OperatorMetatype],
                      layer_attributes: BaseLayerAttributes = None,
                      node_id_override: int = None,
                      layer_name: LayerName = None,
                      ignored_algorithms: List[str] = None,
                      is_in_iteration_scope: bool = False,
                      is_integer_input: bool = False,
                      is_shared: bool = False) -> NNCFNode:
        """
        Adds a node into the graph. A node represents an operation being performed on tensors.
        :param node_name: The name of the node to add - will serve as a human-readable and specifiable ID.
        :param node_type: The type of the node - usually a framework-specific string representation of the operation.
        :param node_metatype: The metatype of the node - a framework-abstract definition of what the operation
            actually means.
        :param layer_attributes: Must be passed for operations that, in order to be processed
            during compression, require additional information such as the exact dimension of the weights tensor to be
            considered a "channel" dimension for per-channel quantization, the weight shape itself for sparsity mask
            creation etc.
        :param node_id_override: The numerical ID to be associated with the new node; if unspecified, will
            assign a unique ID.
        :param layer_name: The name of the framework-specific "layer" object that houses the operation represented by
            the node.
        :param ignored_algorithms: A list of compression algorithm names (from the same set of strings that are
            specified in the `"algorithm": ...` section of the .json NNCF config) which should ignore this operation.
        :param is_in_iteration_scope: Whether the node to be currently added corresponds to an iteration of an RNN
            cycle (where the number of iterations is determined dynamically based on the RNN input shape).
        :param is_integer_input: Only valid for input nodes - whether the input node corresponds to an integer input.
        :param is_shared: Whether the node corresponds to an operation that accesses the weights that are also accessed
            by another operation (represented by a separate node) in NNCFGraph. Examples would be repeated applications
            of a convolution layer with the same weights at different stages in the network.
        :return: An NNCFNode object representing the newly created node in the graph. The node will still have
            to be connected to the rest of the nodes with edges using the `NNCFGraph.add_edge_between_nncf_nodes`
            method.
        """
        if node_id_override is not None:
            node_id = node_id_override
        else:
            node_ids = self.get_all_node_ids()
            if node_ids:
                node_id = max(self.get_all_node_ids()) + 1
            else:
                node_id = 0

        if node_id in self._node_id_to_key_dict:
            raise ValueError(f'NNCF node with id {node_id} is already in the NNCFGraph')

        node_key = f'{node_id} {node_name}'

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            NNCFGraph.ID_NODE_ATTR: node_id,
            NNCFGraph.KEY_NODE_ATTR: node_key,
            NNCFGraph.NODE_NAME_ATTR: node_name,
            NNCFGraph.NODE_TYPE_ATTR: node_type,
            NNCFGraph.LAYER_NAME_ATTR: layer_name,
            NNCFGraph.METATYPE_ATTR: node_metatype,
            NNCFGraph.IS_SHARED_ATTR: is_shared,
            NNCFGraph.IS_IN_ITERATION_SCOPE_NODE_ATTR: is_in_iteration_scope,
            NNCFGraph.IS_INTEGER_INPUT_NODE_ATTR: is_integer_input
        }
        if layer_attributes is not None:
            attrs[NNCFGraph.LAYER_ATTRIBUTES] = layer_attributes

        if ignored_algorithms is None:
            ignored_algorithms = []
        attrs[NNCFGraph.IGNORED_ALGOS_ATTR] = ignored_algorithms
        self._nx_graph.add_node(node_key, **attrs)

        node = NNCFNode(node_id, data=attrs)

        if node.metatype in get_input_metatypes():
            self._input_nncf_nodes[node_id] = node

        if node.metatype in get_output_metatypes():
            self._output_nncf_nodes[node_id] = node

        if layer_name is not None:
            self._node_ids_vs_layer_names[node.node_id] = layer_name
            self._layer_name_vs_shared_nodes[layer_name].append(node)

        return node

    def add_edge_between_nncf_nodes(self, from_node_id: int, to_node_id: int,
                                    tensor_shape: List[int],
                                    input_port_id: int,
                                    dtype: Dtype):
        """
        Adds a directed edge between two `NNCFNode`s that are already present in the graph.
        The edge represents an activation tensor, produced or consumed by an operation (which is represented by a node)
        :param from_node_id: The `NNCFNode.node_id` of the node that produces the tensor represented by the edge.
        :param to_node_id: The `NNCFNode.node_id` of the node that consumes the tensor represented by the edge.
        :param tensor_shape: The shape of the tensor represented by the edge.
        :param input_port_id: Specifies the index among the possible inputs of the `to_node_id` node' that this tensor
            should correspond to.
        :param dtype: The data type of the tensor.
        """
        from_node_key = self._node_id_to_key_dict[from_node_id]
        to_node_key = self._node_id_to_key_dict[to_node_id]

        err_reason = None

        if from_node_key not in self._nx_graph.nodes:
            err_reason = f'node {from_node_key} not in NNCFGraph'
        if to_node_key not in self._nx_graph.nodes:
            err_reason = f'node {from_node_key} not in NNCFGraph'
        if from_node_id in self._output_nncf_nodes:
            err_reason = 'cannot add edges *from* output nodes'
        if to_node_id in self._input_nncf_nodes:
            err_reason = 'cannot add edges *to* input nodes'

        if err_reason is not None:
            raise ValueError(f'Cannot add edge from {from_node_key} to {to_node_key} - {err_reason}!')

        attrs = {
            NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: tensor_shape,
            NNCFGraph.IN_PORT_NAME_EDGE_ATTR: input_port_id,
            NNCFGraph.DTYPE_EDGE_ATTR: dtype
        }
        self._nx_graph.add_edge(from_node_key, to_node_key, **attrs)

    def topological_sort(self) -> List[NNCFNode]:
        """
        Returns nodes in topologically sorted order, additionally sorted in ascending node ID order.
        """
        return [self._nx_node_to_nncf_node(self._nx_graph.nodes[node_name])
                for node_name in
                nx.lexicographical_topological_sort(self._nx_graph,
                                                    key=lambda x: self._nx_graph.nodes[x][NNCFGraph.ID_NODE_ATTR])]

    def dump_graph(self, path: str):
        nx.drawing.nx_pydot.write_dot(self.get_graph_for_structure_analysis(), path)

    def visualize_graph(self, path: str):
        out_graph = self._get_graph_for_visualization()
        nx.drawing.nx_pydot.write_dot(out_graph, path)
        try:
            A = to_agraph(out_graph)
            A.layout('dot')
            png_path = os.path.splitext(path)[0]+'.png'
            A.draw(png_path)
        except ImportError:
            nncf_logger.warning('Graphviz is not installed - only the .dot model visualization format will be used. '
                                'Install pygraphviz into your Python environment and graphviz system-wide to enable '
                                'PNG rendering.')

    def get_graph_for_structure_analysis(self, extended: bool = False) -> nx.DiGraph:
        """
        The graph to dump has certain node attributes omitted, compared to the graph stored
        inside NNCFGraph.

        :param extended - whether the graph should have additional node attributes for improved visualization
        :return: An nx.DiGraph to be used for structure analysis
        """
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            attrs_node = {
                'id': node[NNCFGraph.ID_NODE_ATTR],
                'type': node[NNCFGraph.NODE_TYPE_ATTR]
            }
            if 'color' in node:
                attrs_node['color'] = node['color']
            if 'label' in node:
                attrs_node['label'] = node['label']
            if 'style' in node:
                attrs_node['style'] = node['style']

            out_graph.add_node(node_name, **attrs_node)
        if extended:
            for u, v in self._nx_graph.edges:
                out_graph.add_edge(u, v, label=self._nx_graph.edges[u, v][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
        else:
            for u, v in self._nx_graph.edges:
                out_graph.add_edge(u, v)

        return out_graph

    def _get_graph_for_visualization(self) -> nx.DiGraph:
        """
        :return: A user-friendly graph .dot file, making it easier to debug the network and setup
        ignored/target scopes.
        """
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            attrs_node = {}
            attrs_node['label'] = str(node[NNCFGraph.ID_NODE_ATTR]) + ' ' + str(node[NNCFGraph.NODE_NAME_ATTR])
            out_graph.add_node(node_name, **attrs_node)

        for u, v in self._nx_graph.edges:
            out_graph.add_edge(u, v, label=self._nx_graph.edges[u, v][NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])

        mapping = {k: v['label'] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop('label')

        return out_graph

    def get_node_by_name(self, name: NNCFNodeName) -> NNCFNode:
        matches = []
        for nx_node in self._nx_graph.nodes.values():
            if nx_node[NNCFGraph.NODE_NAME_ATTR] == name:
                matches.append(nx_node)
        if not matches:
            raise RuntimeError('Could not find a node {} in NNCFGraph!'.format(name))
        if len(matches) > 1:
            raise RuntimeError('More than one node in NNCFGraph matches name {}:\n{}'.
                               format(name,
                                      '\t\n'.join(
                                          [n[NNCFGraph.KEY_NODE_ATTR] for n in matches])))
        return self._nx_node_to_nncf_node(next(iter(matches)))

    def __eq__(self, other: 'NNCFGraph'):
        nm = iso.categorical_node_match([NNCFGraph.ID_NODE_ATTR,
                                         NNCFGraph.KEY_NODE_ATTR,
                                         NNCFGraph.LAYER_ATTRIBUTES], [None, None, None])
        em = iso.categorical_edge_match([NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR,
                                         NNCFGraph.IN_PORT_NAME_EDGE_ATTR], [None, None])
        return nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)

    def get_nx_graph_copy(self) -> nx.DiGraph:
        return deepcopy(self._nx_graph)

    def get_matching_nncf_graph_pattern_io_list(self, expression: Expression) -> List[NNCFGraphPatternIO]:
        matched_node_key_sequences = find_subgraphs_matching_expression(self._nx_graph, expression)
        pattern_ios = [self.get_nncf_graph_pattern_io(match) for match in matched_node_key_sequences]
        return pattern_ios

    def get_nncf_graph_pattern_io(self, match: List[str]) -> NNCFGraphPatternIO:
        """
        Returns an NNCFGraphPatternIO object that describes the input/output nodes and edges of a
        subgraph specified by `match`.

        :param match: A list of node keys specifying a subgraph to be matched. The subgraph to be matched will
        consist of nodes with the same keys that are connected with edges in the order they are listed in the
        `match` list
        :return: NNCFGraphPatternIO object describing the inputs and outputs of the matched subgraph
        """
        in_edge_boundary, out_edge_boundary = get_edge_boundaries(match, self._nx_graph)
        boundary = in_edge_boundary + out_edge_boundary
        input_nncf_edges = []
        output_nncf_edges = []

        for nx_edge in boundary:
            from_node_key = nx_edge[0]
            to_node_key = nx_edge[1]
            data = nx_edge[2]
            nncf_edge = NNCFGraphEdge(self._nx_node_to_nncf_node(self._nx_graph.nodes[from_node_key]),
                                      self._nx_node_to_nncf_node(self._nx_graph.nodes[to_node_key]),
                                      data[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR])
            if from_node_key in match:
                output_nncf_edges.append(nncf_edge)
            elif to_node_key in match:
                input_nncf_edges.append(nncf_edge)
            else:
                raise RuntimeError('Invalid graph expression supplied!')

        return NNCFGraphPatternIO(input_nncf_edges, output_nncf_edges)

    def get_nx_edge(self, node_u: NNCFNode, node_v: NNCFNode):
        nx_node_u = self._nx_graph.nodes[self._node_id_to_key_dict[node_u.node_id]]
        nx_node_v = self._nx_graph.nodes[self._node_id_to_key_dict[node_v.node_id]]
        return self._nx_graph.edges[nx_node_u['key'], nx_node_v['key']]

    def get_nodes_count(self):
        return self._nx_graph.number_of_nodes()
