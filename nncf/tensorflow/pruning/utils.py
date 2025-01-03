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

from typing import Dict, List

import numpy as np
import tensorflow as tf

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.logging import nncf_logger
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.keras_layers import TFBatchNormalizationLayerMetatype
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.graph.utils import get_original_name_and_instance_idx
from nncf.tensorflow.graph.utils import unwrap_layer
from nncf.tensorflow.layers.data_layout import get_input_channel_axis
from nncf.tensorflow.layers.data_layout import get_weight_channel_axis
from nncf.tensorflow.layers.wrapper import NNCFWrapper


def get_filter_axis(layer: NNCFWrapper, weight_attr: str) -> int:
    channel_axes = get_weight_channel_axis(layer, weight_attr)
    filter_axis = channel_axes[0] if isinstance(channel_axes, tuple) else channel_axes
    return filter_axis


def get_filters_num(layer: NNCFWrapper):
    layer_metatype = get_keras_layer_metatype(layer)
    if len(layer_metatype.weight_definitions) != 1:
        raise ValueError(f"Could not calculate the number of filters for the layer {layer.layer.name}.")

    weight_def = layer_metatype.weight_definitions[0]
    weight_attr = weight_def.weight_attr_name

    if layer_metatype is TFBatchNormalizationLayerMetatype and not layer.layer.scale:
        nncf_logger.debug(
            "Fused gamma parameter encountered in BatchNormalization layer. "
            "Using beta parameter instead to calculate the number of filters."
        )
        weight_attr = "beta"

    filter_axis = get_filter_axis(layer, weight_attr)
    filters_num = layer.layer_weights[weight_attr].shape[filter_axis]
    return filters_num


def is_valid_shape(shape):
    if shape is None:
        return False
    if None in shape:
        return False
    return True


def broadcast_filter_mask(filter_mask, shape, dim):
    broadcasted_filter_mask = tf.zeros(shape)
    meta_shape = np.ones(len(shape), dtype=np.int64)
    meta_shape[dim] = filter_mask.shape[0]
    broadcasted_filter_mask += tf.reshape(filter_mask, tuple(meta_shape))
    return broadcasted_filter_mask


def collect_output_shapes(model: "NNCFNetwork", graph: NNCFGraph) -> Dict[NNCFNodeName, List[int]]:  # noqa: F821
    """
    Collects output dimension shapes for convolutions and fully connected layers
    from the connected edges in the NNCFGraph.

    :param graph: NNCFGraph.
    :return: Dictionary of output dimension shapes. E.g {node_name: (height, width)}
    """
    layers_out_shapes = {}
    for node in graph.get_nodes_by_metatypes(GENERAL_CONV_LAYER_METATYPES):
        node_name, node_index = get_original_name_and_instance_idx(node.node_name)
        layer = model.get_layer(node_name)
        layer_ = unwrap_layer(layer)

        channel_axis = get_input_channel_axis(layer)
        dims_slice = (
            slice(channel_axis - layer_.rank, channel_axis)
            if layer.data_format == "channels_last"
            else slice(channel_axis + 1, None)
        )
        in_shape = layer.get_input_shape_at(node_index)[dims_slice]
        out_shape = layer.get_output_shape_at(node_index)[dims_slice]

        if not is_valid_shape(in_shape) or not is_valid_shape(out_shape):
            raise nncf.ValidationError(f"Input/output shape is not defined for layer `{layer.name}` ")

        layers_out_shapes[node.node_name] = out_shape

    for node in graph.get_nodes_by_metatypes(LINEAR_LAYER_METATYPES):
        node_name, node_index = get_original_name_and_instance_idx(node.node_name)
        layer = model.get_layer(node_name)

        in_shape = layer.get_input_shape_at(node_index)[1:]
        out_shape = layer.get_output_shape_at(node_index)[1:]

        if not is_valid_shape(in_shape) or not is_valid_shape(out_shape):
            raise nncf.ValidationError(f"Input/output shape is not defined for layer `{layer.name}` ")

        layers_out_shapes[node.node_name] = out_shape
    return layers_out_shapes
