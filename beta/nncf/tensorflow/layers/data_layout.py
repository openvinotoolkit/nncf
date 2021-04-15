"""
 Copyright (c) 2020 Intel Corporation
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

from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper
from beta.nncf.tensorflow.layers.common import ALL_LAYERS_WITH_WEIGHTS
from beta.nncf.tensorflow.layers.common import CHANNEL_AXES
from beta.nncf.tensorflow.layers.common import GENERAL_CONV_LAYERS
from beta.nncf.tensorflow.layers.common import WEIGHT_ATTR_NAME


def get_channel_size(input_shape, input_type, input_name, layer):
    channel_axes = get_channel_axis(input_type, input_name, layer)
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
    original_layer = layer.layer if isinstance(layer, NNCFWrapper) else layer
    data_format = get_data_format(layer)
    class_name = original_layer.__class__.__name__
    if class_name in GENERAL_CONV_LAYERS:
        return -1 if data_format == 'channels_last' else -1 - original_layer.rank
    if class_name in ['BatchNormalization', 'LayerNormalization']:
        return original_layer.axis

    return -1 if data_format == 'channels_last' else 1


def get_weight_channel_axis(layer, weight_attr):
    original_layer = layer.layer if isinstance(layer, NNCFWrapper) else layer
    class_name = original_layer.__class__.__name__
    if class_name in ALL_LAYERS_WITH_WEIGHTS \
            and weight_attr == ALL_LAYERS_WITH_WEIGHTS[class_name][WEIGHT_ATTR_NAME]:
        return ALL_LAYERS_WITH_WEIGHTS[class_name].get(CHANNEL_AXES, -1)
    return -1
