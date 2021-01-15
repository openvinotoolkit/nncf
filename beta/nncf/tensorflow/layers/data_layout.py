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

from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.layers.common import LAYERS_WITH_WEIGHTS


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
    data_format = get_data_format(layer)
    return -1 if data_format == 'channels_last' else 1


def get_weight_channel_axis(layer, weight_attr):
    original_layer = layer.layer if isinstance(layer, NNCFWrapper) else layer
    class_name = original_layer.__class__.__name__
    if class_name in LAYERS_WITH_WEIGHTS and weight_attr == LAYERS_WITH_WEIGHTS[class_name]['weight_attr_name']:
        return LAYERS_WITH_WEIGHTS[class_name].get('channel_axes', -1)
    return -1
