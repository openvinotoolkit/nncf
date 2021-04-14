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

from beta.nncf.tensorflow.layers.common import DECONV_LAYERS
from beta.nncf.tensorflow.layers.common import ALL_LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import WEIGHT_ATTR_NAME
from beta.nncf.tensorflow.layers.data_layout import get_weight_channel_axis
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.module_attributes import ConvolutionModuleAttributes


def is_depthwise_conv(node: NNCFNode) -> bool:
    return isinstance(node.module_attributes, ConvolutionModuleAttributes) \
           and node.module_attributes.groups == node.module_attributes.in_channels \
           and (node.module_attributes.out_channels % node.module_attributes.in_channels == 0) \
           and node.module_attributes.in_channels > 1 \
           or node.node_type == 'DepthwiseConv2D'


def is_conv_with_downsampling(node: NNCFNode) -> bool:
    return isinstance(node.module_attributes, ConvolutionModuleAttributes) \
           and not np.all(np.array(node.module_attributes.stride) == 1) \
           and node.node_type not in DECONV_LAYERS


def is_shared(node: NNCFNode) -> bool:
    return node.data['is_shared']


def get_filter_axis(layer: NNCFWrapper, weight_attr: str) -> int:
    channel_axes = get_weight_channel_axis(layer, weight_attr)
    filter_axis = channel_axes[0] if isinstance(channel_axes, tuple) else channel_axes
    return filter_axis


def get_filters_num(layer: NNCFWrapper):
    layer_type = layer.layer.__class__.__name__
    layer_props = ALL_LAYERS_WITH_WEIGHTS[layer_type]
    weight_attr = layer_props[WEIGHT_ATTR_NAME]

    filter_axis = get_filter_axis(layer, weight_attr)
    filters_num = layer.layer_weights[weight_attr].shape[filter_axis]
    return filters_num


def broadcast_filter_mask(filter_mask, shape, dim):
    broadcasted_filter_mask = tf.zeros(shape)
    meta_shape = np.ones(len(shape), dtype=np.int64)
    meta_shape[dim] = filter_mask.shape[0]
    broadcasted_filter_mask += tf.reshape(filter_mask, tuple(meta_shape))
    return broadcasted_filter_mask


class TFPruningOperationsMetatypeRegistry(PruningOperationsMetatypeRegistry):
    @staticmethod
    def get_version_agnostic_name(name):
        return name


def get_original_layer_type(layer: tf.keras.layers.Layer):
    if isinstance(layer, NNCFWrapper):
        layer = layer.layer
    return type(layer)
