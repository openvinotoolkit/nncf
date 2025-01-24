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

from functools import partial
from typing import List, Set, Tuple, Type, Union

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging import nncf_logger
from nncf.common.pruning.utils import traverse_function


def get_concat_axis(input_shapes: List[List[int]], output_shapes: List[List[int]]) -> int:
    """
    Returns concatenation axis by given input and output shape of concat node.

    :param input_shapes: Input_shapes of given concat node.
    :param output_shapes: Output_shapes of given concat node.
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
            nncf_logger.debug("Identity concat node detected")
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
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in op_types, visited=visited)

    first_nodes_of_type = []
    for root in graph_roots:
        first_nodes_of_type.extend(graph.traverse_graph(root, partial_traverse_function))
    return first_nodes_of_type


def get_split_axis(input_shapes: List[List[int]], output_shapes: List[List[int]]) -> int:
    """
    Returns split/chunk axis by given input and output shape of split/chunk node.

    :param input_shapes: Input_shapes of given split/chunk node.
    :param output_shapes: Output_shapes of given split/chunk node.
    :returns: Split/Chunk axis of given split/chunk node.
    """
    axis = None
    for idx, (dim_in, dim_out) in enumerate(zip(input_shapes[0], output_shapes[0])):
        if dim_in != dim_out:
            axis = idx
            break

    if axis is None:
        axis = -1
        nncf_logger.debug("Identity split/concat node detected")

    return axis


def get_number_of_quantized_ops(
    graph: NNCFGraph,
    quantizer_metatypes: List[Type[OperatorMetatype]],
    quantizable_metatypes: List[Type[OperatorMetatype]],
) -> int:
    """
    Returns the number of quantized operations in the graph.

    :param graph: NNCF graph which was built for the quantized model.
    :param quantizer_metatypes: List of quantizer's metatypes. For example,
        it may be metatype correspondent to FakeQuantize operation or
        metatypes correspondent to Quantize/Dequantize operations.
    :param quantizable_metatypes: List of metatypes for operations
        that may be quantized.
    :return: Number of quantized operations in the graph.
    """
    quantized_ops: Set[NNCFNode] = set()
    nodes_to_see: List[NNCFNode] = []

    for quantizer_node in graph.get_nodes_by_metatypes(quantizer_metatypes):
        nodes_to_see.extend(graph.get_next_nodes(quantizer_node))
        while nodes_to_see:
            node = nodes_to_see.pop()
            if node.metatype in quantizable_metatypes:
                quantized_ops.add(node)
            else:
                nodes_to_see.extend(graph.get_next_nodes(node))
    return len(quantized_ops)


def get_reduction_axes(
    channel_axes: Union[List[int], Tuple[int, ...]], shape: Union[List[int], Tuple[int, ...]]
) -> Tuple[int, ...]:
    """
    Returns filtered reduction axes without axes that correspond to channels.

    :param channel_axes: Channel axes.
    :param shape: Shape that need to be filtered.
    :return: Reduction axes.
    """
    reduction_axes = list(range(len(shape)))
    for channel_axis in sorted(channel_axes, reverse=True):
        del reduction_axes[channel_axis]
    return tuple(reduction_axes)
