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


def discard_zeros(x: np.ndarray) -> np.ndarray:
    x = x[x != 0]
    if x.shape == (0,):
        x = np.array([0])
    return x


def get_per_channel_history(inputs_tensor: np.ndarray, axis: list) -> np.ndarray:
    new_shape = 1
    inputs_tensor_shape = inputs_tensor.shape
    for dim in axis:
        new_shape *= inputs_tensor_shape[dim]
    return np.transpose(np.reshape(inputs_tensor, (new_shape, -1)))


def get_axes(ndims, per_channel, channel_axes, add_dim=False):
    axes = list(range(ndims))
    if per_channel:
        for val in channel_axes:
            val = (ndims + val) % ndims
            axes.remove(val)
    if add_dim:
        # if all input tensors are stacked together - one more dimension
        axes.append(axes[-1] + 1)
    return axes
