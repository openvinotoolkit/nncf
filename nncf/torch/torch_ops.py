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

from typing import Any, Optional, Tuple, TypeVar, Union

import torch

TensorType = TypeVar("TensorType")


def is_tensor(target: Any):
    return isinstance(target, torch.Tensor)


# Tensor functions


def device(target: TensorType) -> torch.device:
    return target.device


def squeeze(target: TensorType, axis: Optional[Union[int, Tuple[int]]] = None) -> TensorType:
    if axis is None:
        return target.squeeze()
    return target.squeeze(axis)


def zeros_like(target: TensorType) -> TensorType:
    return torch.zeros_like(target)


def ones_like(target: TensorType) -> TensorType:
    return torch.ones_like(target)


def count_nonzero(target, axis: Optional[TensorType] = None) -> TensorType:
    return torch.count_nonzero(target, dim=axis)


def max(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    if axis is None:
        return torch.max(target)
    return torch.tensor(torch.max(target, dim=axis).values)


def min(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    if axis is None:
        return torch.min(target)
    return torch.tensor(torch.min(target, dim=axis).values)


def absolute(target: TensorType) -> TensorType:
    return torch.absolute(target)


def maximum(target: TensorType, other: TensorType) -> TensorType:
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=target.data.device)
    return torch.maximum(target, other)


def minimum(target: TensorType, other: TensorType) -> TensorType:
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=target.data.device)
    return torch.minimum(target, other)


def all(target: TensorType) -> TensorType:
    return torch.all(target)


def any(target: TensorType) -> TensorType:
    return torch.any(target)


def where(condition: TensorType, x: TensorType, y: TensorType) -> TensorType:
    return torch.where(condition, x, y)


def is_empty(target: TensorType) -> TensorType:
    return target.numel() == 0
