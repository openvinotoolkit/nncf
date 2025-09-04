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
from typing import Union

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import GenericWeightedLayerAttributes
from nncf.common.graph.layer_attributes import GroupNormLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.logging import nncf_logger
from nncf.common.pruning.utils import traverse_function


def get_concat_axis(input_shapes: list[list[int]], output_shapes: list[list[int]]) -> int:
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


def get_first_nodes_of_type(graph: NNCFGraph, op_types: list[str]) -> list[NNCFNode]:
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


def get_split_axis(input_shapes: list[list[int]], output_shapes: list[list[int]]) -> int:
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
    quantizer_metatypes: list[type[OperatorMetatype]],
    quantizable_metatypes: list[type[OperatorMetatype]],
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
    quantized_ops: set[NNCFNode] = set()
    nodes_to_see: list[NNCFNode] = []

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
    channel_axes: Union[list[int], tuple[int, ...]], shape: Union[list[int], tuple[int, ...]]
) -> tuple[int, ...]:
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


def get_weight_shape_legacy(layer_attributes: WeightedLayerAttributes) -> list[int]:
    """
    Returns hard-coded weights shape layout for the given layer attributes.
    Applicable only for eager PyTorch and Tensorflow models.

    :param layer_attributes: Layer attributes of a NNCFNode.
    :return: Weights shape layout.
    """
    if isinstance(layer_attributes, LinearLayerAttributes):
        return [layer_attributes.out_features, layer_attributes.in_features]

    if isinstance(layer_attributes, ConvolutionLayerAttributes):
        if not layer_attributes.transpose:
            return [
                layer_attributes.out_channels,
                layer_attributes.in_channels // layer_attributes.groups,
                *layer_attributes.kernel_size,
            ]
        return [
            layer_attributes.in_channels,
            layer_attributes.out_channels // layer_attributes.groups,
            *layer_attributes.kernel_size,
        ]

    if isinstance(layer_attributes, GroupNormLayerAttributes):
        return [layer_attributes.num_channels]

    assert isinstance(layer_attributes, GenericWeightedLayerAttributes)
    return layer_attributes.weight_shape


def get_target_dim_for_compression_legacy(layer_attributes: WeightedLayerAttributes) -> int:
    """
    Returns hard-coded target dim for compression for the given layer attributes.
    Applicable only for eager PyTorch and Tensorflow models.

    :param layer_attributes: Layer attributes of a NNCFNode.
    :return: Target dim for compression.
    """
    if isinstance(layer_attributes, ConvolutionLayerAttributes):
        # Always quantize per each "out" channel
        return 1 if layer_attributes.transpose else 0

    else:
        assert isinstance(
            layer_attributes, (GenericWeightedLayerAttributes, LinearLayerAttributes, GroupNormLayerAttributes)
        )
        return 0


def get_bias_shape_legacy(layer_attributes: WeightedLayerAttributes) -> int:
    """
    Returns hard-coded bias shape for the given linear layer attributes.
    Applicable only for eager PyTorch and Tensorflow models.

    :param layer_attributes: Linear layer attributes of a NNCFNode.
    :return: Correspondent bias shape.
    """
    assert isinstance(layer_attributes, LinearLayerAttributes)
    return layer_attributes.out_features if layer_attributes.with_bias is True else 0


def get_num_filters_legacy(layer_attributes: WeightedLayerAttributes) -> int:
    """
    Returns hard-coded number of filters for the given layer attributes.
    Applicable only for eager PyTorch and Tensorflow models.

    :param layer_attributes: Layer attributes of a NNCFNode.
    :return: Correspondent number of filters.
    """
    assert isinstance(layer_attributes, WeightedLayerAttributes)
    weight_shape = get_weight_shape_legacy(layer_attributes)
    return weight_shape[get_target_dim_for_compression_legacy(layer_attributes)]
