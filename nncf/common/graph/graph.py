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
import pathlib
from collections import defaultdict
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Collection,
    Dict,
    Generator,
    KeysView,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    ValuesView,
    cast,
)

import networkx as nx  # type:ignore
import networkx.algorithms.isomorphism as iso  # type:ignore
from networkx.classes.reportviews import OutEdgeView  # type:ignore

import nncf
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import INPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OUTPUT_NOOP_METATYPES
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import GraphPattern
from nncf.common.utils.dot_file_rw import write_dot_graph

NNCFNodeName = str
LayerName = str


class NNCFNode:
    """
    Class describing nodes used in NNCFGraph.
    """

    ID_NODE_ATTR = "id"
    NODE_NAME_ATTR = "node_name"
    KEY_NODE_ATTR = "key"
    NODE_TYPE_ATTR = "type"
    METATYPE_ATTR = "metatype"
    LAYER_NAME_ATTR = "layer_name"
    LAYER_ATTRIBUTES = "layer_attributes"
    IGNORED_ALGOS_ATTR = "ignored_algos"
    IS_IN_ITERATION_SCOPE_NODE_ATTR = "is_in_iteration_scope"
    IS_INTEGER_INPUT_NODE_ATTR = "is_integer_input"
    IS_SHARED_ATTR = "is_shared"

    def __init__(self, attributes: Dict[str, Any]) -> None:
        self._attributes = attributes

    @property
    def attributes(self) -> Dict[str, Any]:
        return self._attributes

    @property
    def node_id(self) -> int:
        return cast(int, self._attributes[NNCFNode.ID_NODE_ATTR])

    @property
    def node_key(self) -> str:
        return cast(str, self._attributes[NNCFNode.KEY_NODE_ATTR])

    @property
    def node_name(self) -> NNCFNodeName:
        return cast(NNCFNodeName, self._attributes[NNCFNode.NODE_NAME_ATTR])

    @property
    def metatype(self) -> Type[OperatorMetatype]:
        return cast(Type[OperatorMetatype], self._attributes[NNCFNode.METATYPE_ATTR])

    @property
    def node_type(self) -> str:
        return cast(str, self._attributes[NNCFNode.NODE_TYPE_ATTR])

    @property
    def layer_name(self) -> Optional[LayerName]:
        return self._attributes.get(NNCFNode.LAYER_NAME_ATTR)

    @layer_name.setter
    def layer_name(self, value: str) -> None:
        self._attributes[NNCFNode.LAYER_NAME_ATTR] = value

    @property
    def layer_attributes(self) -> Optional[BaseLayerAttributes]:
        return self._attributes.get(NNCFNode.LAYER_ATTRIBUTES)

    @layer_attributes.setter
    def layer_attributes(self, value: BaseLayerAttributes) -> None:
        self._attributes[NNCFNode.LAYER_ATTRIBUTES] = value

    @property
    def ignored_algorithms(self) -> List[str]:
        return cast(List[str], self._attributes[NNCFNode.IGNORED_ALGOS_ATTR])

    def is_in_iteration_scope(self) -> bool:
        return cast(bool, self._attributes[NNCFNode.IS_IN_ITERATION_SCOPE_NODE_ATTR])

    def is_integer_input(self) -> bool:
        return cast(bool, self._attributes[NNCFNode.IS_INTEGER_INPUT_NODE_ATTR])

    def is_shared(self) -> bool:
        return cast(bool, self._attributes[NNCFNode.IS_SHARED_ATTR])

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return " ".join([str(self.node_id), self.node_name, self.node_type])

    def __hash__(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NNCFNode) and self.attributes == other.attributes


class NNCFGraphEdge:
    """
    A structure describing an edge in NNCFGraph. Since nodes of the NNCFGraph are operations
    on (activation) tensors, an edge in NNCFGraph is a representation of an activation tensor produced or
    consumed by an operation.
    """

    def __init__(
        self,
        from_node: NNCFNode,
        to_node: NNCFNode,
        input_port_id: int,
        output_port_id: int,
        tensor_shape: List[int],
        dtype: Dtype,
        parallel_input_port_ids: List[int],
    ) -> None:
        """
        :param from_node: An NNCFNode that sources the directed edge.
        :param to_node: An NNCFNode that sinks the directed edge.
        :param input_port_id: The ID of the tensor input to the `to_node` that this edge corresponds to.
        :param output_port_id: The ID of the tensor output of the `from_node` that this edge corresponds to..
        :param tensor_shape: The shape of the activation tensor the edge represents.
        :param dtype: The data type of the activation tensor the edge represents.
        """
        self.from_node = from_node
        self.to_node = to_node
        self.input_port_id = input_port_id
        self.output_port_id = output_port_id
        self.tensor_shape: Tuple[int, ...] = tuple(tensor_shape)
        self.dtype = dtype
        self.parallel_input_port_ids = parallel_input_port_ids

    def __str__(self) -> str:
        return f"{self.from_node}:{self.output_port_id} -> {self.tensor_shape} -> {self.to_node}:{self.input_port_id}"

    def __hash__(self) -> int:
        return hash(
            (
                self.from_node,
                self.to_node,
                self.input_port_id,
                self.output_port_id,
                tuple(self.tensor_shape),
                self.dtype,
                tuple(self.parallel_input_port_ids),
            )
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NNCFGraphEdge) and self.__dict__ == other.__dict__


class NNCFGraphPatternIO:
    """
    Describes the inputs and outputs of a subgraph in NNCFGraph.
    """

    def __init__(self, input_edges: List[NNCFGraphEdge], output_edges: List[NNCFGraphEdge]):
        self.input_edges = input_edges
        self.output_edges = output_edges


class NNCFGraph:
    """
    Wrapper over a regular directed acyclic graph that represents a control flow/execution graph of a DNN
    providing some useful methods for graph traversal.
    """

    ACTIVATION_SHAPE_EDGE_ATTR = "activation_shape"
    INPUT_PORT_ID_EDGE_ATTR = "input_port_id"
    OUTPUT_PORT_ID_EDGE_ATTR = "output_port_id"
    DTYPE_EDGE_ATTR = "dtype"
    PARALLEL_INPUT_PORT_IDS_ATTR = "parallel_input_ports"

    def __init__(self) -> None:
        self._nx_graph = nx.DiGraph()
        self._node_id_to_key_dict: Dict[int, str] = {}
        self._nodes: Dict[str, NNCFNode] = {}
        self._input_nncf_nodes: Dict[int, NNCFNode] = {}
        self._output_nncf_nodes: Dict[int, NNCFNode] = {}
        self._node_ids_vs_layer_names: Dict[int, LayerName] = {}
        self._layer_name_vs_shared_nodes: Dict[LayerName, List[NNCFNode]] = defaultdict(list)
        self._node_name_to_node_id_map: Dict[str, List[int]] = {}

    @property
    def nodes(self) -> Dict[str, NNCFNode]:
        return self._nodes

    def get_node_by_id(self, node_id: int) -> NNCFNode:
        """
        :param node_id: Id of the node.
        :return: Node in a graph with such id.
        """
        return self.get_node_by_key(self.get_node_key_by_id(node_id))

    def get_node_by_key(self, key: str) -> NNCFNode:
        """
        :param key: key (node_name) of the node.
        :return: NNCFNode in a graph with such key.
        """
        return self._nodes[key]

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
        all_nodes_of_type: List[NNCFNode] = []
        if not type_list:
            return all_nodes_of_type
        for nncf_node in self.nodes.values():
            if nncf_node.node_type in type_list:
                all_nodes_of_type.append(nncf_node)
        return all_nodes_of_type

    def get_nodes_by_metatypes(self, metatype_list: Collection[Type[OperatorMetatype]]) -> List[NNCFNode]:
        """
        Return a list of nodes with provided metatypes.

        :param metatype_list: List of types to look for.
        :return: List of nodes with provided metatypes.
        """
        all_nodes_of_type = []
        for nncf_node in self.get_all_nodes():
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
        return list(self._nodes.values())

    def get_all_simple_paths(
        self, start_node_name: NNCFNodeName, end_node_name: NNCFNodeName
    ) -> Generator[List[str], None, None]:
        """
        Generates all simple paths in the NNCFGraph from start node to end node.
        A simple path is a path with no repeated nodes.

        :param start_node_name: a name of starting node for path
        :param end_node_name: a name of node at which to end path
        :return: A generator that produces lists of simple paths. If there are no paths between the start and end nodes
        the generator produces no output.
        """
        start_node = self.get_node_by_name(start_node_name)
        end_node = self.get_node_by_name(end_node_name)
        start_node_key = self.get_node_key_by_id(start_node.node_id)
        end_node_key = self.get_node_key_by_id(end_node.node_id)
        return cast(Generator[List[str], None, None], nx.all_simple_paths(self._nx_graph, start_node_key, end_node_key))

    @staticmethod
    def _get_edge_boundaries(
        match: List[str], graph: nx.DiGraph
    ) -> Tuple[List[Tuple[str, str, Dict[str, Any]]], List[Tuple[str, str, Dict[str, Any]]]]:
        out_edge_boundary = list(nx.edge_boundary(graph, match, data=True))
        complement = list(filter(lambda x: x not in match, graph.nodes.keys()))
        in_edge_boundary = list(nx.edge_boundary(graph, complement, data=True))
        return sorted(in_edge_boundary), sorted(out_edge_boundary)  # must be sorted for determinism

    def get_node_key_by_id(self, node_id: int) -> str:
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
        return [self._nodes[key] for key in nx_node_keys]

    def get_previous_nodes(self, node: NNCFNode) -> List[NNCFNode]:
        """
        Returns producer nodes of provided node.

        :param node: Consumer node.
        :return: List of producers nodes of provided node.
        """

        nx_node_keys = self._nx_graph.pred[self._node_id_to_key_dict[node.node_id]]
        return [self._nodes[key] for key in nx_node_keys]

    def get_input_edges(self, node: NNCFNode) -> List[NNCFGraphEdge]:
        """
        Returns edges of input tensors with description sorted by input port ID.

        :param node: Consumer node.
        :return: List of input edges for the node sorted by input port ID.
        """
        input_nodes = self.get_previous_nodes(node)
        edges = []
        for from_node in input_nodes:
            edges.extend(self._get_edges(from_node, node))
        return sorted(edges, key=lambda x: x.input_port_id)

    def get_input_edge_by_port_id(self, node: NNCFNode, port_id: int) -> NNCFGraphEdge:
        """
        Returns the input edge for a given node, where edge.input_port_id == port_id is True.

        :param node: The node for which to retrieve the input edge.
        :param port_id: The ID of the input port to filter the edges.
        :return: An input edge connected to the specified input port ID of the
            given node.
        """
        edges = [e for e in self.get_input_edges(node) if e.input_port_id == port_id]
        if len(edges) == 0:
            raise nncf.ValidationError(
                f"Node {node.node_name} does not contain input edge connected to {port_id} port ID."
            )

        if len(edges) > 1:
            raise nncf.InternalError(
                "Unsupported graph. More than one edge was found for a given node by the specified input port ID."
            )
        return edges[0]

    def get_output_edges(self, node: NNCFNode) -> List[NNCFGraphEdge]:
        """
        Returns edges of output tensors sorted by output port ID.

        :param node: Producer node.
        :return: List of output edges for the node sorted by output port ID.
        """
        output_nodes = self.get_next_nodes(node)
        edges = []
        for to_node in output_nodes:
            edges.extend(self._get_edges(node, to_node))
        return sorted(edges, key=lambda x: x.output_port_id)

    def get_output_edges_by_port_id(self, node: NNCFNode, port_id: int) -> List[NNCFGraphEdge]:
        """
        Returns a list of output edges for a given node, filtered by the specified
        output port ID (edge.output_port_id == port_id).

        :param node: The node for which to retrieve the output edges.
        :param port_id: The ID of the output port to filter the edges.
        :return: A list of the output edges connected to the specified output port ID
            of the given node.
        """
        return [e for e in self.get_output_edges(node) if e.output_port_id == port_id]

    def _get_edges(self, from_node: NNCFNode, to_node: NNCFNode) -> List[NNCFGraphEdge]:
        edges = []
        edge = self.get_edge(from_node, to_node)
        parallel_input_port_ids = edge.parallel_input_port_ids
        edge.parallel_input_port_ids = []
        edges.append(edge)
        for input_port_id in parallel_input_port_ids:
            edges.append(
                NNCFGraphEdge(
                    from_node=edge.from_node,
                    to_node=edge.to_node,
                    input_port_id=input_port_id,
                    output_port_id=edge.output_port_id,
                    tensor_shape=list(edge.tensor_shape),
                    dtype=edge.dtype,
                    parallel_input_port_ids=[],
                )
            )
        return edges

    def traverse_graph(
        self,
        curr_node: NNCFNode,
        traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
        traverse_forward: bool = True,
    ) -> List[Any]:
        """
        Traverses graph up or down starting form `curr_node` node.

        :param curr_node: Node from which traversal is started.
        :param traverse_function: Function describing condition of traversal continuation/termination.
        :param traverse_forward: Flag specifying direction of traversal.
        :return:
        """
        output: List[Any] = []
        return self._traverse_graph_recursive_helper(curr_node, traverse_function, output, traverse_forward)

    def _traverse_graph_recursive_helper(
        self,
        curr_node: NNCFNode,
        traverse_function: Callable[[NNCFNode, List[Any]], Tuple[bool, List[Any]]],
        output: List[Any],
        traverse_forward: bool,
    ) -> List[Any]:
        is_finished, output = traverse_function(curr_node, output)
        get_nodes_fn = self.get_next_nodes if traverse_forward else self.get_previous_nodes
        if not is_finished:
            for node in get_nodes_fn(curr_node):
                self._traverse_graph_recursive_helper(node, traverse_function, output, traverse_forward)
        return output

    def add_nncf_node(
        self,
        node_name: str,
        node_type: str,
        node_metatype: Type[OperatorMetatype],
        layer_attributes: Optional[BaseLayerAttributes] = None,
        node_id_override: Optional[int] = None,
        layer_name: Optional[LayerName] = None,
        ignored_algorithms: Optional[List[str]] = None,
        is_in_iteration_scope: bool = False,
        is_integer_input: bool = False,
        is_shared: bool = False,
    ) -> NNCFNode:
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
            the node and associated trainable weights, if any.
        :param ignored_algorithms: A list of compression algorithm names (from the same set of strings that are
            specified in the `"algorithm": ...` section of the .json NNCF config or `ptq_quantization`)
            which should ignore this operation.
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
            node_ids = list(self.get_all_node_ids())
            if node_ids:
                node_id = max(self.get_all_node_ids()) + 1
            else:
                node_id = 0

        if node_id in self._node_id_to_key_dict:
            raise ValueError(f"NNCF node with id {node_id} is already in the NNCFGraph")

        node_ids = self._node_name_to_node_id_map.setdefault(node_name, [])
        node_ids.append(node_id)

        node_key = f"{node_id} {node_name}"

        self._node_id_to_key_dict[node_id] = node_key
        attrs = {
            NNCFNode.ID_NODE_ATTR: node_id,
            NNCFNode.NODE_NAME_ATTR: node_name,
            NNCFNode.KEY_NODE_ATTR: node_key,
            NNCFNode.NODE_TYPE_ATTR: node_type,
            NNCFNode.LAYER_NAME_ATTR: layer_name,
            NNCFNode.METATYPE_ATTR: node_metatype,
            NNCFNode.IS_SHARED_ATTR: is_shared,
            NNCFNode.IS_IN_ITERATION_SCOPE_NODE_ATTR: is_in_iteration_scope,
            NNCFNode.IS_INTEGER_INPUT_NODE_ATTR: is_integer_input,
        }
        if layer_attributes is not None:
            attrs[NNCFNode.LAYER_ATTRIBUTES] = layer_attributes

        if ignored_algorithms is None:
            ignored_algorithms = []
        attrs[NNCFNode.IGNORED_ALGOS_ATTR] = ignored_algorithms
        self._nx_graph.add_node(node_key, **attrs)

        node = NNCFNode(self._nx_graph.nodes[node_key])
        self._nodes[node_key] = node

        if node.metatype in INPUT_NOOP_METATYPES:
            self._input_nncf_nodes[node_id] = node

        if node.metatype in OUTPUT_NOOP_METATYPES:
            self._output_nncf_nodes[node_id] = node

        if layer_name is not None:
            self._node_ids_vs_layer_names[node.node_id] = layer_name
            self._layer_name_vs_shared_nodes[layer_name].append(node)

        return node

    def add_edge_between_nncf_nodes(
        self,
        from_node_id: int,
        to_node_id: int,
        tensor_shape: Union[Tuple[int, ...], List[int]],
        input_port_id: int,
        output_port_id: int,
        dtype: Dtype,
        parallel_input_port_ids: Optional[List[int]] = None,
    ) -> None:
        """
        Adds a directed edge between two `NNCFNode`s that are already present in the graph.
        The edge represents an activation tensor, produced or consumed by an operation (which is represented by a node)
        :param from_node_id: The `NNCFNode.node_id` of the node that produces the tensor represented by the edge.
        :param to_node_id: The `NNCFNode.node_id` of the node that consumes the tensor represented by the edge.
        :param tensor_shape: The shape of the tensor represented by the edge.
        :param input_port_id: Specifies the index among the possible inputs of the `to_node_id` node' that this tensor
            should correspond to.
        :param output_port_id: Specifies the index among the possible outputs of the `from_node_id` node' that this
            tensor should correspond to.
        :param dtype: The data type of the tensor.
        :param parallel_input_port_ids: Input ports for parallel edges, if any should be present for this edge.
        """
        from_node_key = self._node_id_to_key_dict[from_node_id]
        to_node_key = self._node_id_to_key_dict[to_node_id]

        err_reason = None

        if from_node_key not in self._nx_graph.nodes:
            err_reason = f"node {from_node_key} not in NNCFGraph"
        if to_node_key not in self._nx_graph.nodes:
            err_reason = f"node {from_node_key} not in NNCFGraph"
        if from_node_id in self._output_nncf_nodes:
            err_reason = "cannot add edges *from* output nodes"
        if to_node_id in self._input_nncf_nodes:
            err_reason = "cannot add edges *to* input nodes"

        if err_reason is not None:
            raise ValueError(f"Cannot add edge from {from_node_key} to {to_node_key} - {err_reason}!")

        attrs = {
            NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR: tensor_shape,
            NNCFGraph.INPUT_PORT_ID_EDGE_ATTR: input_port_id,
            NNCFGraph.OUTPUT_PORT_ID_EDGE_ATTR: output_port_id,
            NNCFGraph.DTYPE_EDGE_ATTR: dtype,
            NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR: [] if parallel_input_port_ids is None else parallel_input_port_ids,
        }
        self._nx_graph.add_edge(from_node_key, to_node_key, **attrs)

    def topological_sort(self) -> List[NNCFNode]:
        """
        Returns nodes in topologically sorted order, additionally sorted in ascending node ID order.
        """
        return [
            self._nodes[node_name]
            for node_name in nx.lexicographical_topological_sort(
                self._nx_graph, key=lambda x: self._nx_graph.nodes[x][NNCFNode.ID_NODE_ATTR]
            )
        ]

    def dump_graph(self, path: str) -> None:
        write_dot_graph(self.get_graph_for_structure_analysis(), pathlib.Path(path))

    def visualize_graph(self, path: str) -> None:
        out_graph = self._get_graph_for_visualization()
        write_dot_graph(out_graph, pathlib.Path(path))

    def get_graph_for_structure_analysis(self, extended: bool = False) -> nx.DiGraph:
        """
        Returns the nx.Digraph, which is built based on self._nx_graph.
        The new graph has certain node attributes omitted, compared to the graph stored inside NNCFGraph.
        If the node name consists of a special reserved character, this character will be replaced.

        :param extended: whether the graph edges should have attributes: shape of the tensor and tensor primitive type.
        :return: An nx.DiGraph to be used for structure analysis
        """
        out_graph = nx.DiGraph()
        for node_name, node in self._nx_graph.nodes.items():
            attrs_node = {"id": str(node[NNCFNode.ID_NODE_ATTR]), "type": node[NNCFNode.NODE_TYPE_ATTR]}
            for attr in ["color", "label", "style"]:
                if attr in node:
                    attrs_node[attr] = node[attr]

            out_graph.add_node(node_name, **attrs_node)

        for u, v in self._nx_graph.edges:
            edge = self._nx_graph.edges[u, v]
            attrs_edge = {}
            label = {}
            if edge[NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR]:
                label["parallel_input_port_ids"] = edge[NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR]

            if extended:
                if edge[NNCFGraph.DTYPE_EDGE_ATTR] is Dtype.INTEGER:
                    attrs_edge["style"] = "dashed"
                else:
                    attrs_edge["style"] = "solid"
                label["shape"] = edge[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]

            if label:
                if "shape" in label and len(label) == 1:
                    attrs_edge["label"] = label["shape"]
                else:
                    attrs_edge["label"] = ", ".join((f"{k} {v}" for k, v in label.items()))
            out_graph.add_edge(u, v, **attrs_edge)
        return out_graph

    def _get_graph_for_visualization(self) -> nx.DiGraph:
        """
        :return: A user-friendly graph .dot file, making it easier to debug the network and setup
        ignored/target scopes.
        """
        out_graph = nx.DiGraph()
        for node in self.get_all_nodes():
            attrs_node = {}
            attrs_node["label"] = f"{node.node_id} {node.node_name}"
            node_key = self.get_node_key_by_id(node.node_id)
            out_graph.add_node(node_key, **attrs_node)

        for u, v in self._nx_graph.edges:
            edge = self._nx_graph.edges[u, v]
            if edge[NNCFGraph.DTYPE_EDGE_ATTR] is Dtype.INTEGER:
                style = "dashed"
            else:
                style = "solid"
            edge_label = (
                f"{edge[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR]} \\n"
                f"{edge[NNCFGraph.OUTPUT_PORT_ID_EDGE_ATTR]} -> {edge[NNCFGraph.INPUT_PORT_ID_EDGE_ATTR]}"
            )
            out_graph.add_edge(u, v, label=edge_label, style=style)

        mapping = {k: v["label"] for k, v in out_graph.nodes.items()}
        out_graph = nx.relabel_nodes(out_graph, mapping)
        for node in out_graph.nodes.values():
            node.pop("label")  # type: ignore

        return out_graph

    def get_node_by_name(self, name: NNCFNodeName) -> NNCFNode:
        node_ids = self._node_name_to_node_id_map.get(name, None)
        if node_ids is None:
            raise nncf.InternalError("Could not find a node {} in NNCFGraph!".format(name))
        if len(node_ids) > 1:
            raise nncf.InternalError(f"More than one node in NNCFGraph matches name {name}")

        node_key = f"{node_ids[0]} {name}"
        return self._nodes[node_key]

    def __eq__(self, other: object) -> bool:
        nm = iso.categorical_node_match(
            [NNCFNode.ID_NODE_ATTR, NNCFNode.KEY_NODE_ATTR, NNCFNode.LAYER_ATTRIBUTES], [None, None, None]
        )
        em = iso.categorical_edge_match(
            [NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR, NNCFGraph.INPUT_PORT_ID_EDGE_ATTR], [None, None]
        )
        return isinstance(other, NNCFGraph) and bool(
            nx.is_isomorphic(self._nx_graph, other._nx_graph, node_match=nm, edge_match=em)
        )

    def get_nx_graph_copy(self) -> nx.DiGraph:
        return deepcopy(self._nx_graph)

    def get_nncf_graph_pattern_io(self, match: List[str]) -> NNCFGraphPatternIO:
        """
        Returns an NNCFGraphPatternIO object that describes the input/output nodes and edges of a
        subgraph specified by `match`.

        :param match: A list of node keys specifying a subgraph to be matched. The subgraph to be matched will
        consist of nodes with the same keys that are connected with edges in the order they are listed in the
        `match` list
        :return: NNCFGraphPatternIO object describing the inputs and outputs of the matched subgraph
        """

        in_edge_boundary, out_edge_boundary = NNCFGraph._get_edge_boundaries(match, self._nx_graph)
        boundary = in_edge_boundary + out_edge_boundary
        input_nncf_edges = []
        output_nncf_edges = []

        for nx_edge in boundary:
            from_node_key = nx_edge[0]
            to_node_key = nx_edge[1]
            data = nx_edge[2]
            nncf_edge = NNCFGraphEdge(
                self._nodes[from_node_key],
                self._nodes[to_node_key],
                input_port_id=data[NNCFGraph.INPUT_PORT_ID_EDGE_ATTR],
                output_port_id=data[NNCFGraph.OUTPUT_PORT_ID_EDGE_ATTR],
                tensor_shape=data[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR],
                dtype=data[NNCFGraph.DTYPE_EDGE_ATTR],
                parallel_input_port_ids=data[NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR],
            )
            if from_node_key in match:
                output_nncf_edges.append(nncf_edge)
            elif to_node_key in match:
                input_nncf_edges.append(nncf_edge)
            else:
                raise nncf.InternalError("Invalid graph expression supplied!")

        return NNCFGraphPatternIO(input_nncf_edges, output_nncf_edges)

    def get_nx_edge(self, node_u: NNCFNode, node_v: NNCFNode) -> OutEdgeView:
        nx_node_u = self._nx_graph.nodes[self._node_id_to_key_dict[node_u.node_id]]
        nx_node_v = self._nx_graph.nodes[self._node_id_to_key_dict[node_v.node_id]]
        return self._nx_graph.edges[nx_node_u["key"], nx_node_v["key"]]

    def get_nodes_count(self) -> int:
        return int(self._nx_graph.number_of_nodes())

    def get_edge(self, from_node: NNCFNode, to_node: NNCFNode) -> NNCFGraphEdge:
        """
        Returns an NNCFGraphEdge object that corresponds to an edge connecting two given NNCFNodes in this
        graph.
        :param from_node: The NNCFNode in this graph that sources the edge.
        :param to_node: The NNCFNode in this graph that sinks the edge.
        :return: The NNCFGraphEdge object representing the edge between `from_node` and `to_node`.
        """
        data = self.get_nx_edge(from_node, to_node)
        return NNCFGraphEdge(
            from_node,
            to_node,
            data[NNCFGraph.INPUT_PORT_ID_EDGE_ATTR],
            data[NNCFGraph.OUTPUT_PORT_ID_EDGE_ATTR],
            data[NNCFGraph.ACTIVATION_SHAPE_EDGE_ATTR],
            data[NNCFGraph.DTYPE_EDGE_ATTR],
            data[NNCFGraph.PARALLEL_INPUT_PORT_IDS_ATTR],
        )

    def get_all_edges(self) -> Generator[NNCFGraphEdge, None, None]:
        for nx_edge in self._nx_graph.in_edges:
            yield self.get_edge(self.get_node_by_key(nx_edge[0]), self.get_node_by_key(nx_edge[1]))

    def remove_nodes_from(self, nodes: Collection[NNCFNode]) -> None:
        """
        Removes nodes from the current NNCFGraph instance.
        We use the remove_node method here because remove_nodes_from uses a silent fail instead of an exception.

        :param nodes: List of NNCFNodes to remove.
        """
        for node in nodes:
            self._nx_graph.remove_node(node.node_key)
            del self._nodes[node.node_key]

        self._node_id_to_key_dict = {}
        for node_key, node in self._nx_graph.nodes.items():
            self._node_id_to_key_dict[node["id"]] = node_key  # type:ignore

    def find_matching_subgraphs(self, patterns: GraphPattern, strict: bool = True) -> List[List[NNCFNode]]:
        """
        Returns subgraphs of matched pattern in patterns.

        :param patterns: Instance of GraphPattern containing all patterns.
        :param strict: If True returns only strict matched subgraphs, if False - all matched subgraphs.
        :return: List of subgraphs that are matching by pattern matching.
            Subgraph is a ordered list of nodes of matched subgraph
        The returned nodes order relies on DiGraphMatcher isomorphic subgraphs matching logic from networkX package.
        DiGraphMatcher does not guarantee a specific order for returning isomorphic subgraphs.
        """
        output = []
        for matched_subgraph in find_subgraphs_matching_pattern(self._nx_graph, patterns, strict):
            subgraph_list = []
            for node_key in matched_subgraph:
                subgraph_list.append(self.get_node_by_key(node_key))
            output.append(subgraph_list)
        return output
