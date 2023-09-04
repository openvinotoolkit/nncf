# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import deque
from copy import deepcopy
from typing import Callable, Deque, List, Tuple, Union

from nncf.common.quantization.structs import QuantizerScaleShape
from nncf.common.tensor import NNCFTensor


def get_channel_count_and_dim_idx(scale_shape: Tuple[int]) -> Tuple[int, int]:
    channel_dim_idx = 0
    channel_count = 1
    for dim_idx, dim in enumerate(scale_shape):
        if dim != 1:
            channel_dim_idx = dim_idx
            channel_count = dim
    return channel_count, channel_dim_idx


def split_into_channels(input_: NNCFTensor, scale_shape: Tuple[int]) -> List[NNCFTensor]:
    channel_count, channel_dim_idx = get_channel_count_and_dim_idx(scale_shape)
    backend = input_.backend
    channel_first_tensor = backend.moveaxis(input_, channel_dim_idx, 0)
    if channel_count == 1:
        return [channel_first_tensor]

    ret_list = []
    for i in range(channel_count):
        ret_list.append(channel_first_tensor[i, ...])
    return ret_list


def get_per_channel_history(
    raw_input_history: Deque[NNCFTensor], reduction_shape: Tuple[int], discard_zeros: bool = False
) -> List[NNCFTensor]:
    channel_count, _ = get_channel_count_and_dim_idx(reduction_shape)
    per_channel_history = [None for _ in range(channel_count)]
    if not raw_input_history:
        return per_channel_history

    backend = next(iter(raw_input_history)).backend
    for _ in range(len(raw_input_history)):
        entry = raw_input_history.popleft()
        split = split_into_channels(entry, reduction_shape)
        for i in range(channel_count):
            flat_channel_split = split[i].flatten()

            if discard_zeros:
                # For post-RELU quantizers exact zeros may prevail and lead to
                # zero mean and MAD - discard them
                flat_channel_split = flat_channel_split[flat_channel_split != 0]

            if per_channel_history[i] is None:
                per_channel_history[i] = flat_channel_split
            else:
                per_channel_history[i] = backend.concatenate([per_channel_history[i], flat_channel_split])
        raw_input_history.append(entry)
    return per_channel_history


ReductionAxes = Tuple[int]


def percentile_reduce(input_: NNCFTensor, reduction_axes: ReductionAxes, pc: float) -> NNCFTensor:
    backend = input_.backend
    quantile = pc / 100
    return backend.quantile(input_, quantile, axis=list(reduction_axes), keepdims=True)


ReductionShape = Tuple[int]
REDUCE_TO_SCALAR_REDUCTION_SHAPE = (-1,)


def get_reduction_axes_from_scale_shape(scale_shape: QuantizerScaleShape, channel_idx: int = None) -> ReductionAxes:
    if scale_shape.is_per_tensor():
        return REDUCE_TO_SCALAR_REDUCTION_SHAPE
    if channel_idx is not None:
        return tuple(i for i in range(len(scale_shape.shape)) if i != channel_idx)
    return tuple(i for i, dim in enumerate(scale_shape.shape) if dim == 1)


def is_reduce_to_scalar(reduction_axes: ReductionAxes) -> bool:
    return reduction_axes == REDUCE_TO_SCALAR_REDUCTION_SHAPE


def get_reduction_shape_from_sample_shape(sample_shape: List[int], reduction_axes: ReductionAxes) -> ReductionShape:
    if is_reduce_to_scalar(reduction_axes):
        return (1,)
    reduced_shape = deepcopy(list(sample_shape))
    for ax in reduction_axes:
        reduced_shape[ax] = 1
    return tuple(reduced_shape)


MaskedReduceFN = Callable[[NNCFTensor, Union[int, tuple, list], NNCFTensor, bool], NNCFTensor]
