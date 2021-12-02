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

from functools import partial
from typing import List

from nncf.common.graph import NNCFGraph, NNCFNode
from nncf.common.pruning.utils import traverse_function

from nncf.common.utils.logger import logger


def get_concat_axis(input_shapes: List[List[int]], output_shapes: List[List[int]]) -> int:
    """
    Returns concatenation axis by given input and output shape of concat node.

    :param input_shapes: Input_shapes of given concat node.
    :param output_shapes: Input_shapes of given concat node.
    :returns: Concatenation axis of given concat node.
    """
    axis = None
    none_dim = None
    for idx, (dim_in, dim_out) in enumerate(zip(input_shapes[0], output_shapes[0])):
        if dim_in != dim_out:
            axis = idx
            break
        if dim_in is None:
            none_dim = idx

    if axis is None:
        if none_dim is None:
            axis = -1
            logger.warning('Identity concat node detected')
        else:
            axis = none_dim

    return axis


def get_first_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for first node in graph with type in `op_types`.
    First == layer with type in `op_types`, that there is a path from the input such that there are no other
    operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all first nodes with type in `op_types`.
    """
    graph_roots = graph.get_input_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function,
                                        type_check_fn=lambda x: x in op_types,
                                        visited=visited)

    first_nodes_of_type = []
    for root in graph_roots:
        first_nodes_of_type.extend(graph.traverse_graph(root, partial_traverse_function))
    return first_nodes_of_type
