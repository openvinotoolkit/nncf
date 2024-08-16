# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
from enum import Enum
from typing import Dict, List, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype

TModel = TypeVar("TModel")


class NodeMarker(Enum):
    """
    Markers are used to classify graph nodes.

    - Node marker `DEPENDS_ON_INPUTS` means that the node depends on input data (nodes that are
      designated as graph inputs).
    - Node marker `CONSTANT_SUBGRAPH` means that the node is part of a constant subgraph, i.e.,
      its output tensor does not depend on any input data.
    """

    DEPENDS_ON_INPUT = "DEPENDS_ON_INPUT"
    CONSTANT_SUBGRAPH = "CONSTANT_SUBGRAPH"


def classify_nodes(
    graph: NNCFGraph,
    start_nodes: List[NNCFNode],
    constant_metatypes: List[OperatorMetatype],
) -> Dict[NNCFNode, NodeMarker]:
    """
    Assigns a marker to each node in the input graph.

    :param graph: The input graph to be analyzed.
    :param start_nodes: A list of nodes designated as graph inputs. These nodes are used to identify
        which nodes depend on input data.
    :param constant_metatypes: A list of metatypes representing backend-specific constant operations.
    :return: A dictionary mapping each node to its assigned marker.
    """
    # NOTE: All nodes in the graph that have no input edges should either be
    # in `start_nodes` or have a constant metatype.

    # TODO(andrey-churkin): It seems possible to identify a ShapeOf subgraph using
    # this approach. Suppose we have a marker `SHAPEOF_SUBGRAPH`. If a node has this marker,
    # it means that the node is part of a ShapeOf subgraph. Naturally, all nodes with the
    # ShapeOf metatype have this marker. Next, for each node, if all input edges have integer type
    # and all producers are marked using `SHAPEOF_SUBGRAPH` marker, then we assign the `SHAPEOF_SUBGRAPH`
    # marker to the node.

    sorted_nodes = graph.topological_sort()

    node_markers: Dict[NNCFNode, NodeMarker] = {}
    for node in sorted_nodes:
        producer_nodes = graph.get_previous_nodes(node)
        if producer_nodes:
            # NOTE: If there are producer nodes, the `node` marker is determined using the markers
            # of the producer nodes according to the following rule: If all producer nodes
            # have the `CONSTANT_SUBGRAPH` marker, then the marker for `node` is also `CONSTANT_SUBGRAPH`.
            # Otherwise, the marker for node is `DEPENDS_ON_INPUT`.

            # NOTE: It is guaranteed that all producers already have their markers because the nodes
            # are considered in topological order.
            node_markers[node] = (
                NodeMarker.CONSTANT_SUBGRAPH
                if all(node_markers[v] == NodeMarker.CONSTANT_SUBGRAPH for v in producer_nodes)
                else NodeMarker.DEPENDS_ON_INPUT
            )
        elif node in start_nodes:
            node_markers[node] = NodeMarker.DEPENDS_ON_INPUT
        elif node.metatype in constant_metatypes:
            node_markers[node] = NodeMarker.CONSTANT_SUBGRAPH
        else:
            raise ValueError(
                "The input graph contains a node {node} that does not have input edges but is not classified "
                " as a constant node or a start node."
            )

    return node_markers


def remove_constant_subgraphs(
    graph: NNCFGraph,
    initial_node_markers: Dict[NNCFNode, NodeMarker],
    current_node_markers: Dict[NNCFNode, NodeMarker],
) -> NNCFGraph:
    """
    :param graph:
    :param initial_node_markers:
    :param current_node_markers:
    :return:
    """
    constant_subgraph_nodes = set()

    node_to_input_port_ids: Dict[NNCFNode, List[int]] = {}
    for node in graph.get_all_nodes():
        # Collect all nodes with `CONSTANT_SUBGRAPH` marker
        if current_node_markers[node] == NodeMarker.CONSTANT_SUBGRAPH:
            constant_subgraph_nodes.add(node)

        for e in graph.get_input_edges(node):
            initial_marker = initial_node_markers[e.from_node]
            current_marker = current_node_markers[e.from_node]
            if initial_marker == NodeMarker.DEPENDS_ON_INPUT and current_marker == NodeMarker.CONSTANT_SUBGRAPH:
                node_to_input_port_ids.setdefault(node, []).append(e.input_port_id)

    preserved_nodes = set(node_to_input_port_ids)
    queue = collections.deque(node_to_input_port_ids)
    while queue:
        node = queue.pop()
        port_ids = node_to_input_port_ids.get(node)
        if port_ids:
            producers = [graph.get_input_edge_by_port_id(node, port_id).from_node for port_id in port_ids]
        else:
            producers = graph.get_previous_nodes(node)

        for v in producers:
            if v not in preserved_nodes:
                queue.append(v)
                preserved_nodes.add(v)

    constant_subgraph_nodes = constant_subgraph_nodes.difference(preserved_nodes)
    graph.remove_nodes_from(constant_subgraph_nodes)

    return graph


def transform_graph(
    graph: NNCFGraph,
    start_nodes: List[NNCFNode],
    shapeof_metatypes: List[OperatorMetatype],
    dropout_metatypes: List[OperatorMetatype],
    constant_metatypes: List[OperatorMetatype],
) -> NNCFGraph:
    """
    Applies the following ordered list of transformations to the provided graph:
        1. Removes all shapeof subgraphs.
        2. Removes constant subgraphs while preserving some of them.
        3. Removes all dropout operations and connects their producers with consumers.
           It assumes that each dropout node has only one producer and one consumer.

    :param graph: The input graph to be transformed.
    :param start_nodes: A list of nodes designated as graph inputs. These nodes are used to identify
        which nodes depend on input data.
    :param shapeof_metatypes: A list of metatypes representing backend-specific shapeof operations.
    :param dropout_metatypes: A list of metatypes representing backend-specific dropout operations.
    :param constant_metatypes: A list of metatypes representing backend-specific constant operations.
    :return: A transformed graph.
    """
    initial_node_markers = classify_nodes(graph, start_nodes, constant_metatypes)
    remove_shapeof_subgraphs(graph, shapeof_metatypes, start_nodes)
    current_node_markers = classify_nodes(graph, start_nodes, constant_metatypes)
    remove_constant_subgraphs(graph, initial_node_markers, current_node_markers)
    remove_nodes_and_reconnect_graph(graph, dropout_metatypes)
    return graph


def transform_to_inference_graph(
    nncf_graph: NNCFGraph,
    input_nodes: List[NNCFNode],
    shapeof_metatypes: List[OperatorMetatype],
    dropout_metatypes: List[OperatorMetatype],
) -> NNCFGraph:
    """
    This method contains inplace pipeline of the passes that uses to provide inference graph without constant flows.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param input_nodes: List of input nodes for the given NNCFGraph.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :param dropout_metatypes: List of backend-specific Dropout metatypes.
    :return: NNCFGraph in the inference style.
    """
    remove_shapeof_subgraphs(nncf_graph, shapeof_metatypes, input_nodes)
    filter_constant_nodes(nncf_graph, input_nodes)
    remove_nodes_and_reconnect_graph(nncf_graph, dropout_metatypes)
    return nncf_graph


def remove_shapeof_subgraphs(
    nncf_graph: NNCFGraph,
    shapeof_metatypes: List[OperatorMetatype],
    input_nodes: List[NNCFNode],
) -> NNCFGraph:
    """
    Removes the ShapeOf subgraphs from the provided NNCFGraph instance inplace.
    Constant subgraph should be already removed from the given NNCFGraph.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :param input_nodes: List of input nodes for the given NNCFGraph.
    :return: NNCFGraph without ShapeOf subgraphs.
    """
    nodes_to_drop = set()
    shape_of_nodes = []
    infer_nodes = []

    nodes_queue = collections.deque(input_nodes)
    while nodes_queue:
        node = nodes_queue.pop()
        if node.metatype in shapeof_metatypes:
            shape_of_nodes.append(node)
            continue
        if node.node_name in infer_nodes:
            continue
        infer_nodes.append(node.node_name)
        nodes_queue.extend(nncf_graph.get_next_nodes(node))

    for shape_of_node in shape_of_nodes:
        nodes_to_drop.add(shape_of_node.node_name)

        shape_of_queue = collections.deque(nncf_graph.get_next_nodes(shape_of_node))
        while shape_of_queue:
            node = shape_of_queue.pop()
            all_output_edges_integer = all(e.dtype == Dtype.INTEGER for e in nncf_graph.get_output_edges(node))
            if node.node_name in nodes_to_drop or node.node_name in infer_nodes or not all_output_edges_integer:
                continue
            nodes_to_drop.add(node.node_name)
            # traverse forward and backward to exclude full shape of subgraph
            # recursion excluded due to infer_nodes list around subgraph shape
            shape_of_queue.extend(nncf_graph.get_next_nodes(node) + nncf_graph.get_previous_nodes(node))

    nncf_graph.remove_nodes_from([nncf_graph.get_node_by_name(name) for name in nodes_to_drop])
    return nncf_graph


def remove_nodes_and_reconnect_graph(
    nncf_graph: NNCFGraph,
    metatypes: List[OperatorMetatype],
) -> NNCFGraph:
    """
    Removes nodes with metatypes specified by `metatypes` parameter from
    the provided NNCFGraph instance and connects previous node of a matched node
    with next nodes of a matched node inplace for each matched node.
    Matched nodes should have only one input node and only one output port.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param metatypes: List of backend-specific metatypes.
    :return: Resulting NNCFGraph.
    """
    if not metatypes:
        return nncf_graph

    nodes_to_drop = []
    for node in nncf_graph.get_nodes_by_metatypes(metatypes):
        if node.metatype in metatypes:
            nodes_to_drop.append(node)

            prev_nodes = nncf_graph.get_previous_nodes(node)
            input_edges = nncf_graph.get_input_edges(node)
            assert len(prev_nodes) == len(input_edges) == 1
            prev_node = prev_nodes[0]
            input_edge = input_edges[0]
            assert not input_edge.parallel_input_port_ids

            # nncf_graph.get_next_edges is not used to preserve
            # parallel_input_port_ids
            for output_node in nncf_graph.get_next_nodes(node):
                output_edge = nncf_graph.get_edge(node, output_node)
                # Connects previous node with all next nodes
                # to keep NNCFGraph connected.
                assert input_edge.dtype == output_edge.dtype
                assert input_edge.tensor_shape == output_edge.tensor_shape
                nncf_graph.add_edge_between_nncf_nodes(
                    from_node_id=prev_node.node_id,
                    to_node_id=output_edge.to_node.node_id,
                    tensor_shape=input_edge.tensor_shape,
                    input_port_id=output_edge.input_port_id,
                    output_port_id=input_edge.output_port_id,
                    dtype=input_edge.dtype,
                    parallel_input_port_ids=output_edge.parallel_input_port_ids,
                )
    nncf_graph.remove_nodes_from(nodes_to_drop)
    return nncf_graph


def filter_constant_nodes(
    nncf_graph: NNCFGraph,
    input_nodes: List[NNCFNode],
) -> NNCFGraph:
    """
    Removes all Constant nodes from NNCFGraph inplace, making it inference graph.
    The traversing starts from the input nodes and nodes with weights.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param input_nodes: List of input nodes for the given NNCFGraph.
    :return: NNCFGraph without Constant nodes.
    """
    if not input_nodes:
        return nncf_graph

    visited_nodes = set()
    nodes_queue = collections.deque(input_nodes)
    while nodes_queue:
        node = nodes_queue.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        nodes_queue.extend(nncf_graph.get_next_nodes(node))
    constant_nodes = [node for node in nncf_graph.get_all_nodes() if node not in visited_nodes]
    nncf_graph.remove_nodes_from(constant_nodes)
    return nncf_graph
