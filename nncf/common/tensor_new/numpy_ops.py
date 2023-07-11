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

import numpy as np

TensorType = TypeVar("TensorType")


def is_tensor(target: Any):
    return isinstance(target, np.ndarray)


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


def device(target: TensorType) -> None:
    return None


def size(target: TensorType, axis: Optional[int] = None) -> TensorType:
    return np.size(target, axis=axis)


def squeeze(target: TensorType, axis: Optional[Union[int, Tuple[int]]] = None) -> TensorType:
    return np.squeeze(target, axis=axis)


def zeros_like(target: TensorType) -> TensorType:
    return np.zeros_like(target)


def ones_like(target: TensorType) -> TensorType:
    return np.ones_like(target)


def count_nonzero(target, axis: Optional[TensorType] = None) -> TensorType:
    return np.count_nonzero(target, axis=axis)


def max(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    return np.max(target, axis=axis)


def min(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    return np.min(target, axis=axis)


def absolute(target: TensorType) -> TensorType:
    return np.absolute(target)


def maximum(target: TensorType, other: TensorType) -> TensorType:
    return np.maximum(target, other)


def minimum(target: TensorType, other: TensorType) -> TensorType:
    return np.minimum(target, other)


def all(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    return np.all(target, axis=axis)


def any(target: TensorType, axis: Optional[TensorType] = None) -> TensorType:
    return np.any(target, axis=axis)


def where(
    condition: np.ndarray,
    x: Union[np.ndarray, float, bool],
    y: Union[np.ndarray, float, bool],
) -> np.ndarray:
    return np.where(condition, x, y)


def allclose(
    a: np.ndarray,
    b: np.ndarray,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    equal_nan: bool = False,
) -> np.ndarray:
    return np.allclose(a=a, b=b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def flatten(target: np.ndarray) -> np.ndarray:
    return target.flatten()


def is_empty(target: np.ndarray) -> TensorType:
    return target.size == 0
