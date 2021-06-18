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

from collections import deque
from typing import List, Tuple

import torch
import numpy as np


def max_reduce_like(input_: torch.Tensor, ref_tensor_shape: List[int]) -> torch.Tensor:
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        retval = input_.max()
        for _ in ref_tensor_shape:
            retval.unsqueeze_(-1)
        return retval
    tmp_max = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            tmp_max, _ = torch.max(tmp_max, dim_idx, keepdim=True)
    return tmp_max


def min_reduce_like(input_: torch.Tensor, ref_tensor_shape: List[int]):
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        retval = input_.min()
        for _ in ref_tensor_shape:
            retval.unsqueeze_(-1)
        return retval
    tmp_min = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            tmp_min, _ = torch.min(tmp_min, dim_idx, keepdim=True)
    return tmp_min


def percentile_reduce_like(input_: torch.Tensor, ref_tensor_shape: List[int], q: float):
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        return torch.from_numpy(np.array([np.percentile(input_.cpu().numpy(), q)])).to(dtype=torch.float)
    tmp = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            numpy_tmp = np.percentile(tmp.cpu().numpy(), q, axis=dim_idx, keepdims=True)
            tmp = torch.from_numpy(numpy_tmp).to(dtype=torch.float)
    return tmp


def get_channel_count_and_dim_idx(scale_shape: List[int]) -> Tuple[int, int]:
    channel_dim_idx = 0
    channel_count = 1
    for dim_idx, dim in enumerate(scale_shape):
        if dim != 1:
            channel_dim_idx = dim_idx
            channel_count = dim
    return channel_count, channel_dim_idx


def expand_like(input_: torch.Tensor, scale_shape: List[int]) -> torch.Tensor:
    retval = input_
    count, idx = get_channel_count_and_dim_idx(scale_shape)
    assert input_.numel() == count
    assert len(input_.size()) == 1
    for _ in range(0, idx):
        retval = retval.unsqueeze(0)
    for _ in range(idx + 1, len(scale_shape)):
        retval = retval.unsqueeze(-1)
    return retval


def split_into_channels(input_: np.ndarray, scale_shape: List[int]) -> List[np.ndarray]:
    channel_count, channel_dim_idx = get_channel_count_and_dim_idx(scale_shape)
    channel_first_tensor = np.moveaxis(input_, channel_dim_idx, 0)
    if channel_count == 1:
        return [channel_first_tensor]

    ret_list = []
    for i in range(channel_count):
        ret_list.append(channel_first_tensor[i, ...])
    return ret_list


def get_per_channel_history(raw_input_history: deque, scale_shape: List[int], discard_zeros=False) -> List:
    channel_count, _ = get_channel_count_and_dim_idx(scale_shape)
    per_channel_history = [None for i in range(channel_count)]
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
