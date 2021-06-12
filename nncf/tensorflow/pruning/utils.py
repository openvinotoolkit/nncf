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

import numpy as np
import tensorflow as tf

from nncf.tensorflow.graph.metatypes.common import DECONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import DEPTHWISE_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.layers.data_layout import get_weight_channel_axis
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes


def is_depthwise_conv(node: NNCFNode) -> bool:
    return node.metatype in DEPTHWISE_CONV_LAYER_METATYPES


def is_conv_with_downsampling(node: NNCFNode) -> bool:
    return isinstance(node.layer_attributes, ConvolutionLayerAttributes) \
           and not np.all(np.array(node.layer_attributes.stride) == 1) \
           and node.metatype not in DECONV_LAYER_METATYPES


def is_shared(node: NNCFNode) -> bool:
    return node.data['is_shared']


def get_filter_axis(layer: NNCFWrapper, weight_attr: str) -> int:
    channel_axes = get_weight_channel_axis(layer, weight_attr)
    filter_axis = channel_axes[0] if isinstance(channel_axes, tuple) else channel_axes
    return filter_axis


def get_filters_num(layer: NNCFWrapper):
    layer_metatype = get_keras_layer_metatype(layer)
    if len(layer_metatype.weight_definitions) != 1:
        raise ValueError(f'Could not calculate the number of filters '
                         f'for the layer {layer.layer.name}.')

    weight_def = layer_metatype.weight_definitions[0]
    weight_attr = weight_def.weight_attr_name

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
