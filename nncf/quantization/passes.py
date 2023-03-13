"""
 Copyright (c) 2023 Intel Corporation
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

import collections
from typing import List

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import OperatorMetatype


def transform_to_inference_graph(nncf_graph: NNCFGraph, shapeof_metatypes: List[OperatorMetatype]) -> NNCFGraph:
    """
    This method contains pipeline of the passes that uses to provide inference graph without constant flows.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :return: NNCFGraph in the inference style.
    """
    inference_nncf_graph = remove_shapeof_subgraphs(nncf_graph, shapeof_metatypes)
    return inference_nncf_graph


def remove_shapeof_subgraphs(nncf_graph: NNCFGraph, shapeof_metatypes: List[OperatorMetatype]) -> NNCFGraph:
    """
    Removes the ShapeOf subgraphs from the provided NNCFGraph instance.

    :param nncf_graph: NNCFGraph instance for the transformation.
    :param shapeof_metatypes: List of backend-specific ShapeOf metatypes.
    :return: NNCFGraph without ShapeOf subgraphs.
    """
    nodes_to_drop = set()
    shape_of_nodes = []
    infer_nodes = []

    nodes_queue = collections.deque(nncf_graph.get_input_nodes())
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
