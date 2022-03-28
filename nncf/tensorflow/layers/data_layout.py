"""
 Copyright (c) 2022 Intel Corporation
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

import tensorflow as tf

from nncf.tensorflow.graph.metatypes.common import ALL_LAYER_METATYPES_WITH_WEIGHTS
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import NORMALIZATION_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.matcher import get_keras_layer_metatype
from nncf.tensorflow.graph.utils import unwrap_layer
from nncf.tensorflow.layers.operation import InputType


def get_channel_size(input_shape, channel_axes):
    if not isinstance(channel_axes, (list, tuple)):
        channel_axes = [channel_axes]
    size = 1
    for axis in channel_axes:
        size *= input_shape[axis]
    return size


def get_channel_axis(input_type, input_name, layer):
    if input_type == InputType.INPUTS:
        return get_input_channel_axis(layer)
    return get_weight_channel_axis(layer, input_name)


def get_data_format(layer):
    return getattr(layer, 'data_format', 'channels_last')


def get_input_channel_axis(layer):
    original_layer = unwrap_layer(layer)
    layer_metatype = get_keras_layer_metatype(original_layer, determine_subtype=False)
    data_format = get_data_format(original_layer)
    if layer_metatype in GENERAL_CONV_LAYER_METATYPES:
        return -1 if data_format == 'channels_last' else -1 - original_layer.rank
    if layer_metatype in NORMALIZATION_LAYER_METATYPES:
        return original_layer.axis

    return -1 if data_format == 'channels_last' else 1


def get_weight_channel_axis(layer, weight_attr):
    original_layer = unwrap_layer(layer)
    layer_metatype = get_keras_layer_metatype(original_layer, determine_subtype=False)
    if layer_metatype in ALL_LAYER_METATYPES_WITH_WEIGHTS:
        for weight_def in layer_metatype.weight_definitions:
            if weight_def.weight_attr_name == weight_attr:
                return weight_def.channel_axes

    return -1

def get_weight_shape(layer: tf.keras.layers.Layer, weight_attr: str) -> tf.TensorShape:
    original_layer = unwrap_layer(layer)
    weight = getattr(original_layer, weight_attr)
    return weight.shape
