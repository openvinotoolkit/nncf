# Copyright (c) 2023 Intel Corporation
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
from typing import List, Optional, TypeVar

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OperatorMetatype

TModel = TypeVar("TModel")


def transform_to_inference_graph(
    nncf_graph: NNCFGraph,
    shapeof_metatypes: List[OperatorMetatype],
    dropout_metatypes: List[OperatorMetatype],
    read_variable_metatypes: Optional[List[OperatorMetatype]] = None,
) -> NNCFGraph:
    """
    This method contains inplace pipeline of the passes that uses to provide inference graph without constant flows.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :param dropout_metatypes: List of backend-specific Dropout metatypes.
    :param read_variable_metatypes: List of backend-specific metatypes
        that also can be interpreted as inputs (ReadValue).
    :return: NNCFGraph in the inference style.
    """
    remove_shapeof_subgraphs(nncf_graph, shapeof_metatypes, read_variable_metatypes)
    remove_nodes_and_reconnect_graph(nncf_graph, dropout_metatypes)
    filter_constant_nodes(nncf_graph, read_variable_metatypes)
    return nncf_graph


def remove_shapeof_subgraphs(
    nncf_graph: NNCFGraph,
    shapeof_metatypes: List[OperatorMetatype],
    read_variable_metatypes: Optional[List[OperatorMetatype]] = None,
) -> NNCFGraph:
    """
    Removes the ShapeOf subgraphs from the provided NNCFGraph instance inplace.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :param read_variable_metatypes: List of backend-specific metatypes
        that also can be interpreted as inputs (ReadValue).
    :return: NNCFGraph without ShapeOf subgraphs.
    """
    read_variable_metatypes = read_variable_metatypes if read_variable_metatypes else []
    nodes_to_drop = set()
    shape_of_nodes = []
    infer_nodes = []

    similar_inputs = nncf_graph.get_nodes_by_metatypes(read_variable_metatypes)
    nodes_queue = collections.deque(nncf_graph.get_input_nodes() + similar_inputs)
    while nodes_queue:
        node = nodes_queue.pop()
        if node.metatype in shapeof_metatypes:
            shape_of_nodes.append(node)
            continue
        if node in infer_nodes:
            continue
        infer_nodes.append(node)
        nodes_queue.extend(nncf_graph.get_next_nodes(node))

    for shape_of_node in shape_of_nodes:
        nodes_to_drop.add(shape_of_node)

        shape_of_queue = collections.deque()
        shape_of_queue.extend(nncf_graph.get_next_nodes(shape_of_node))
        while shape_of_queue:
            node = shape_of_queue.pop()
            if node in nodes_to_drop or node in infer_nodes:
                continue
            nodes_to_drop.add(node)
            # traverse forward and backward to exclude full shape of subgraph
            # recursion excluded due to infer_nodes list around subgraph shape
            shape_of_queue.extend(nncf_graph.get_next_nodes(node) + nncf_graph.get_previous_nodes(node))

    nncf_graph.remove_nodes_from(nodes_to_drop)
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
    nncf_graph: NNCFGraph, read_variable_metatypes: Optional[List[OperatorMetatype]] = None
) -> NNCFGraph:
    """
    Removes all Constant nodes from NNCFGraph inplace, making it inference graph.
    The traversing starts from the input nodes and nodes with weights.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param read_variable_metatypes: List of backend-specific metatypes
        that also can be interpreted as inputs (ReadValue).
    :return: NNCFGraph without Constant nodes.
    """
    read_variable_metatypes = read_variable_metatypes if read_variable_metatypes else []
    input_nodes = nncf_graph.get_input_nodes()
    similar_input_nodes = nncf_graph.get_nodes_by_metatypes(read_variable_metatypes)

    start_nodes = input_nodes + similar_input_nodes

    if not start_nodes:
        return nncf_graph

    visited_nodes = set()
    nodes_queue = collections.deque(start_nodes)
    while nodes_queue:
        node = nodes_queue.pop()
        if node in visited_nodes:
            continue
        visited_nodes.add(node)
        nodes_queue.extend(nncf_graph.get_next_nodes(node))
    constant_nodes = [node for node in nncf_graph.get_all_nodes() if node not in visited_nodes]
    nncf_graph.remove_nodes_from(constant_nodes)
    return nncf_graph
