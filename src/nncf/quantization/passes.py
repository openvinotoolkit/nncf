# Copyright (c) 2026 Intel Corporation
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
from typing import TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import Dtype
from nncf.common.graph.operator_metatypes import OperatorMetatype

TModel = TypeVar("TModel")


def transform_to_inference_graph(
    nncf_graph: NNCFGraph,
    input_nodes: list[NNCFNode],
    shapeof_metatypes: list[type[OperatorMetatype]],
    noop_metatypes: list[type[OperatorMetatype]],
    preserved_metatypes: list[type[OperatorMetatype]],
) -> NNCFGraph:
    """
    This method contains inplace pipeline of the passes that uses to provide inference graph without constant flows.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param input_nodes: List of input nodes for the given NNCFGraph.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :param noop_metatypes: List of backend-specific NoOp metatypes that should be bypassed.
    :param preserved_metatypes: List of backend-specific metatypes that require preserving
        float subgraphs when removing the ShapeOf subgraph.
    :return: NNCFGraph in the inference style.
    """
    shapeof_subgraphs = find_shapeof_subgraphs(nncf_graph, shapeof_metatypes, input_nodes)
    preserved_nodes = find_preserved_nodes(nncf_graph, shapeof_subgraphs, preserved_metatypes)
    constant_subgraphs = find_constant_subgraphs(nncf_graph, input_nodes)

    nodes_to_drop = set([*shapeof_subgraphs, *constant_subgraphs]).difference(preserved_nodes)
    nncf_graph.remove_nodes_from(nodes_to_drop)

    bypass_noop_operation_nodes(nncf_graph, noop_metatypes)
    return nncf_graph


def find_shapeof_subgraphs(
    nncf_graph: NNCFGraph,
    shapeof_metatypes: list[type[OperatorMetatype]],
    input_nodes: list[NNCFNode],
) -> list[NNCFNode]:
    """
    Returns a list of nodes belonging to ShapeOf subgraphs.

    :param nncf_graph: The input graph to be analyzed.
    :param shapeof_metatypes: A list of metatypes representing backend-specific
        ShapeOf operations.
    :param input_nodes: A list of nodes designated as graph inputs. These nodes are
        used to identify which nodes depend on input data.
    :return: A list of nodes belonging to ShapeOf subgraphs.
    """
    shapeof_subgraphs = set()
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
        shapeof_subgraphs.add(shape_of_node)

        shape_of_queue: collections.deque[NNCFNode] = collections.deque()
        shape_of_queue.extend(nncf_graph.get_next_nodes(shape_of_node))
        while shape_of_queue:
            node = shape_of_queue.pop()
            if node in shapeof_subgraphs or node.node_name in infer_nodes:
                continue
            shapeof_subgraphs.add(node)
            # traverse forward and backward to exclude full shape of subgraph
            # recursion excluded due to infer_nodes list around subgraph shape
            shape_of_queue.extend(nncf_graph.get_next_nodes(node) + nncf_graph.get_previous_nodes(node))

    return list(shapeof_subgraphs)


def find_preserved_nodes(
    graph: NNCFGraph,
    shapeof_subgraphs: list[NNCFNode],
    preserved_metatypes: list[type[OperatorMetatype]],
) -> list[NNCFNode]:
    """
    :param graph: The input graph to be analyzed.
    :param shapeof_subgraphs: A list of nodes belonging to ShapeOf subgraphs.
    :param preserved_metatypes: Backend-specific metatypes that require preserving
        float subgraphs when removing the ShapeOf subgraph.
    :return: A list of nodes in float subgraphs of ShapeOf subgraphs.
    """
    preserved_nodes = set()
    for node in graph.get_nodes_by_metatypes(preserved_metatypes):
        for e in graph.get_input_edges(node):
            if e.from_node in shapeof_subgraphs and e.dtype == Dtype.FLOAT:
                preserved_nodes.add(e.from_node)

    queue = collections.deque(preserved_nodes)
    while queue:
        node = queue.pop()

        for e in graph.get_input_edges(node):
            if e.from_node in preserved_nodes:
                continue

            if e.dtype == Dtype.FLOAT and e.from_node in shapeof_subgraphs:
                queue.append(e.from_node)
                preserved_nodes.add(e.from_node)

    return list(preserved_nodes)


def find_constant_subgraphs(
    nncf_graph: NNCFGraph,
    input_nodes: list[NNCFNode],
) -> list[NNCFNode]:
    """
    Returns a list of nodes belonging to constant subgraphs.

    :param nncf_graph: The input graph to be analyzed.
    :param input_nodes: A list of nodes designated as graph inputs. These nodes are
        used to identify which nodes depend on input data.
    :return: A list of nodes belonging to constant subgraphs.
    """
    if not input_nodes:
        return []

    visited_nodes = set()
    nodes_queue = collections.deque(input_nodes)
    while nodes_queue:
        node = nodes_queue.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        nodes_queue.extend(nncf_graph.get_next_nodes(node))
    constant_nodes = [node for node in nncf_graph.get_all_nodes() if node not in visited_nodes]

    return constant_nodes


def _can_bypass_noop_node(node: NNCFNode, graph: NNCFGraph) -> bool:
    """
    Checks that the node can be bypassed by connecting its predecessors directly to its successors.
        - The node must have exactly one input edge.
        - All output edges must have the same dtype and tensor shape as the input edge.

    :param node: The node to check.
    :param graph: The graph containing the node.
    :return: True if the node can be bypassed, False otherwise.
    """
    in_edges = graph.get_input_edges(node)
    out_edges = graph.get_output_edges(node)
    if len(in_edges) != 1:
        # Support only one input edge
        return False
    if any(edge.dtype != in_edges[0].dtype or edge.tensor_shape != in_edges[0].tensor_shape for edge in out_edges):
        # All edges should have the same dtype and tensor shape to bypass the node
        return False
    return True


def bypass_noop_operation_nodes(graph: NNCFGraph, metatypes: list[type[OperatorMetatype]]) -> None:
    """
    Bypasses noop nodes in the graph by connecting their predecessors directly
    to their successors and removing the noop nodes.

    :param graph: The NNCFGraph to be modified.
    """
    noop_nodes = graph.get_nodes_by_metatypes(metatypes)
    for noop_node in noop_nodes:
        if not _can_bypass_noop_node(noop_node, graph):
            continue
        graph.bypass_node(noop_node)
