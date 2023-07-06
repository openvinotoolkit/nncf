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

from typing import Optional, Tuple, TypeVar, Union

import numpy as np

try:
    import torch
except ImportError:
    torch = None

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")


def add(target: TensorType, other: TensorType) -> TensorType:
    return target + other


def radd(target: TensorType, other: TensorType) -> TensorType:
    return other + target


def sub(target: TensorType, other: TensorType) -> TensorType:
    return target - other


def rsub(target: TensorType, other: TensorType) -> TensorType:
    return other - target


def mul(target: TensorType, other: TensorType) -> TensorType:
    return target * other


def rmul(target: TensorType, other: TensorType) -> TensorType:
    return other * target


def pow(target: TensorType, other: TensorType) -> TensorType:
    return target**other


def truediv(target: TensorType, other: TensorType) -> TensorType:
    return target / other


def floordiv(target: TensorType, other: TensorType) -> TensorType:
    return target // other


def neg(target: TensorType) -> TensorType:
    return -target


# Comparison operators


def lt(target: TensorType, other: TensorType) -> TensorType:
    return target < other


def le(target: TensorType, other: TensorType) -> TensorType:
    return target < other


def eq(target: TensorType, other: TensorType) -> TensorType:
    return target == other


def nq(target: TensorType, other: TensorType) -> TensorType:
    return target != other


def gt(target: TensorType, other: TensorType) -> TensorType:
    return target > other


def ge(target: TensorType, other: TensorType) -> TensorType:
    return target >= other


# Tensor functions


def device(target: TensorType) -> Optional[DeviceType]:
    if torch is not None and isinstance(target, torch.Tensor):
        return target.device
    else:
        return None


def size(target: TensorType, axis: Optional[int] = None) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        if axis is None:
            return torch.tensor(target.size())
        return torch.tensor(target.size(dim=axis))
    elif isinstance(target, np.ndarray):
        if axis is None:
            return np.array(target.shape)
        return np.array(target.shape[axis])
    elif isinstance(target, list):
        return len(size)

    raise NotImplemented


def squeeze(target: TensorType, axis: Optional[Union[int, Tuple[int]]] = None) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        if axis is None:
            return target.squeeze()
        return target.squeeze(axis)
    elif isinstance(target, np.ndarray):
        return target.squeeze(axis)

    raise NotImplemented


def zeros_like(target: TensorType) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        return torch.zeros_like(target)
    elif isinstance(target, np.ndarray):
        return np.zeros_like(target)

    raise NotImplemented


def count_nonzero(target, axis: Optional[TensorType] = None) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        return torch.count_nonzero(target, dim=axis)
    elif isinstance(target, np.ndarray):
        return np.count_nonzero(target, axis=axis)

    raise NotImplemented


def maximum(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        if axis is None:
            return torch.max(target)
        return torch.tensor(torch.max(target, dim=axis).values)
    elif isinstance(target, np.ndarray):
        return np.max(target, axis=axis)

    raise NotImplemented


def minimum(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        if axis is None:
            return torch.min(target)
        return torch.tensor(torch.min(target, dim=axis).values)
    elif isinstance(target, np.ndarray):
        return np.min(target, axis=axis)

    raise NotImplemented


def absolute(target: TensorType) -> TensorType:
    if torch is not None and isinstance(target, torch.Tensor):
        return torch.absolute(target)
    elif isinstance(target, np.ndarray):
        return np.abs(target)

    raise NotImplemented
