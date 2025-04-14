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

from typing import Any, Deque, List, Tuple

import numpy as np
from numpy.typing import NDArray


def get_channel_count_and_dim_idx(scale_shape: List[int]) -> Tuple[int, int]:
    channel_dim_idx = 0
    channel_count = 1
    for dim_idx, dim in enumerate(scale_shape):
        if dim != 1:
            channel_dim_idx = dim_idx
            channel_count = dim
    return channel_count, channel_dim_idx


def split_into_channels(input_: NDArray[Any], scale_shape: List[int]) -> List[NDArray[Any]]:
    channel_count, channel_dim_idx = get_channel_count_and_dim_idx(scale_shape)
    channel_first_tensor = np.moveaxis(input_, channel_dim_idx, 0)
    if channel_count == 1:
        return [channel_first_tensor]

    ret_list = []
    for i in range(channel_count):
        ret_list.append(channel_first_tensor[i, ...])
    return ret_list


def get_per_channel_history(
    raw_input_history: Deque[Any], scale_shape: List[int], discard_zeros: bool = False
) -> List[Any]:
    channel_count, _ = get_channel_count_and_dim_idx(scale_shape)
    per_channel_history: List[Any] = [None for i in range(channel_count)]
    for _ in range(len(raw_input_history)):
        entry = raw_input_history.popleft()
        split = split_into_channels(entry, scale_shape)
        for i in range(channel_count):
            flat_channel_split = split[i].flatten()

            if discard_zeros:
                # For post-RELU quantizers exact zeros may prevail and lead to
                # zero mean and MAD - discard them
                flat_channel_split = flat_channel_split[flat_channel_split != 0]

            if per_channel_history[i] is None:
                per_channel_history[i] = flat_channel_split
            else:
                per_channel_history[i] = np.concatenate([per_channel_history[i], flat_channel_split])
        raw_input_history.append(entry)
    return per_channel_history


def np_percentile_reduce_like(input_: NDArray[Any], ref_tensor_shape: Tuple[int], q: float) -> NDArray[Any]:
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        return np.array([np.percentile(input_, q)])
    tmp = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            numpy_tmp = np.percentile(tmp, q, axis=dim_idx, keepdims=True)
            tmp = numpy_tmp
    return tmp
