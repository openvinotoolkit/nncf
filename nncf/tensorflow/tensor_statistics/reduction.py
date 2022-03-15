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

from typing import Union, Tuple

import tensorflow as tf

from nncf.tensorflow.layers.data_layout import get_weight_shape
from nncf.common.tensor_statistics.collectors import ReductionShape


def get_axes(ndims: int, per_channel: bool, channel_axes: Union[int, list, tuple]) -> list:
    axes = list(range(ndims))
    if per_channel:
        for val in channel_axes:
            val = (ndims + val) % ndims
            axes.remove(val)
    return axes


def get_reduction_shape_activations(layer: tf.keras.layers.Layer,
                                    channel_axes: Union[int, tuple, list],
                                    use_per_sample_stats: bool) -> ReductionShape:
    ndims = len(layer.get_input_shape_at(0))
    channel_axes_ = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
    reduction_shape = get_axes(ndims, layer.per_channel, channel_axes_)
    if use_per_sample_stats:
        reduction_shape = reduction_shape[1:]
    return tuple(reduction_shape)


def get_reduction_shape_weights(layer: tf.keras.layers.Layer,
                                weight_attr: str, channel_axes: Union[int, tuple, list],
                                per_channel: bool) -> ReductionShape:
    weight_shape = get_weight_shape(layer, weight_attr)
    ndims = len(weight_shape)
    channel_axes_ = channel_axes if isinstance(channel_axes, (list, tuple)) else [channel_axes]
    reduction_shape = get_axes(ndims, per_channel, channel_axes_)
    return tuple(reduction_shape)


def convert_rs_to_pt_type(input_shape: Tuple[int], reduction_shape: ReductionShape) -> ReductionShape:
    if len(reduction_shape) == len(input_shape):
        pt_reduction_shape = [1]
    else:
        pt_reduction_shape = []
        for dim_idx, dim in enumerate(input_shape):
            if dim_idx in reduction_shape:
                pt_reduction_shape.append(1)
            else:
                pt_reduction_shape.append(dim)
    return tuple(pt_reduction_shape)
