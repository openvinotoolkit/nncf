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

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np

from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric as numeric
from nncf.tensor.functions.dispatcher import register_numpy_types
from nncf.tensor.tensor import TTensor

DTYPE_MAP = {
    TensorDataType.float16: np.dtype(np.float16),
    TensorDataType.float32: np.dtype(np.float32),
    TensorDataType.float64: np.dtype(np.float64),
    TensorDataType.int8: np.dtype(np.int8),
    TensorDataType.int32: np.dtype(np.int32),
    TensorDataType.int64: np.dtype(np.int64),
    TensorDataType.uint8: np.dtype(np.uint8),
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


def validate_device(device: TensorDeviceType) -> None:
    if device is not None and device != TensorDeviceType.CPU:
        raise ValueError("numpy_numeric only supports CPU device.")


def convert_to_numpy_dtype(dtype: TensorDataType) -> np.dtype:
    return DTYPE_MAP[dtype] if dtype is not None else None


@register_numpy_types(numeric.device)
def _(a: Union[np.ndarray, np.generic]) -> TensorDeviceType:
    return TensorDeviceType.CPU


@register_numpy_types(numeric.backend)
def _(a: Union[np.ndarray, np.generic]) -> TensorBackend:
    return TensorBackend.numpy


@register_numpy_types(numeric.squeeze)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[np.ndarray, np.generic]:
    return np.squeeze(a, axis=axis)


@register_numpy_types(numeric.flatten)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return a.flatten()


@register_numpy_types(numeric.max)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.array(np.max(a, axis=axis, keepdims=keepdims))


@register_numpy_types(numeric.min)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> Union[np.ndarray, np.generic]:
    return np.array(np.min(a, axis=axis, keepdims=keepdims))


@register_numpy_types(numeric.abs)
def _(a: Union[np.ndarray, np.generic]) -> Union[np.ndarray, np.generic]:
    return np.absolute(a)


@register_numpy_types(numeric.astype)
def _(a: Union[np.ndarray, np.generic], dtype: TensorDataType) -> Union[np.ndarray, np.generic]:
    return a.astype(DTYPE_MAP[dtype])


@register_numpy_types(numeric.dtype)
def _(a: Union[np.ndarray, np.generic]) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@register_numpy_types(numeric.reshape)
def _(a: Union[np.ndarray, np.generic], shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return a.reshape(shape)


@register_numpy_types(numeric.all)
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.all(a, axis=axis))


@register_numpy_types(numeric.allclose)
def _(
    a: Union[np.ndarray, np.generic],
    b: Union[np.ndarray, np.generic, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@register_numpy_types(numeric.any)
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.any(a, axis=axis))


@register_numpy_types(numeric.count_nonzero)
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.count_nonzero(a, axis=axis))


@register_numpy_types(numeric.isempty)
def _(a: Union[np.ndarray, np.generic]) -> bool:
    return a.size == 0


@register_numpy_types(numeric.isclose)
def _(
    a: Union[np.ndarray, np.generic],
    b: Union[np.ndarray, np.generic, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> np.ndarray:
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@register_numpy_types(numeric.maximum)
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.maximum(x1, x2)


@register_numpy_types(numeric.minimum)
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.minimum(x1, x2)


@register_numpy_types(numeric.ones_like)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.ones_like(a)


@register_numpy_types(numeric.where)
def _(
    condition: Union[np.ndarray, np.generic],
    x: Union[np.ndarray, np.generic, float],
    y: Union[np.ndarray, np.generic, float],
) -> np.ndarray:
    return np.where(condition, x, y)


@register_numpy_types(numeric.zeros_like)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.zeros_like(a)


@register_numpy_types(numeric.stack)
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return np.stack(x, axis=axis)


@register_numpy_types(numeric.concatenate)
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return np.concatenate(x, axis=axis)


@register_numpy_types(numeric.unstack)
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return [np.squeeze(e, axis) for e in np.split(x, x.shape[axis], axis=axis)]


@register_numpy_types(numeric.moveaxis)
def _(a: np.ndarray, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return np.moveaxis(a, source, destination)


@register_numpy_types(numeric.mean)
def _(
    a: Union[np.ndarray, np.generic],
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
    dtype: Optional[TensorDataType] = None,
) -> np.ndarray:
    dtype = convert_to_numpy_dtype(dtype)
    return np.array(np.mean(a, axis=axis, keepdims=keepdims, dtype=dtype))


@register_numpy_types(numeric.median)
def _(
    a: Union[np.ndarray, np.generic],
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> np.ndarray:
    return np.array(np.median(a, axis=axis, keepdims=keepdims))


@register_numpy_types(numeric.round)
def _(a: Union[np.ndarray, np.generic], decimals: int = 0) -> np.ndarray:
    return np.round(a, decimals=decimals)


@register_numpy_types(numeric.power)
def _(a: Union[np.ndarray, np.generic], exponent: Union[np.ndarray, float]) -> Union[np.ndarray, np.generic]:
    return np.power(a, exponent)


@register_numpy_types(numeric.quantile)
def _(
    a: Union[np.ndarray, np.generic],
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
) -> Union[np.ndarray, np.generic]:
    return np.array(np.quantile(a.astype(np.float64, copy=False), q=q, axis=axis, keepdims=keepdims))


@register_numpy_types(numeric.percentile)
def _(
    a: np.ndarray,
    q: Union[float, List[float]],
    axis: Union[int, Tuple[int, ...], List[int]],
    keepdims: bool = False,
) -> List[Union[np.ndarray, np.generic]]:
    return np.quantile(
        a.astype(np.float64, copy=False), q=np.true_divide(np.array(q), 100), axis=axis, keepdims=keepdims
    )


@register_numpy_types(numeric._binary_op_nowarn)
def _(
    a: Union[np.ndarray, np.generic], b: Union[np.ndarray, np.generic, float], operator_fn: Callable
) -> Union[np.ndarray, np.generic]:
    # Run operator with disabled warning
    with np.errstate(invalid="ignore", divide="ignore"):
        return operator_fn(a, b)


@register_numpy_types(numeric._binary_reverse_op_nowarn)
def _(
    a: Union[np.ndarray, np.generic], b: Union[np.ndarray, np.generic, float], operator_fn: Callable
) -> Union[np.ndarray, np.generic]:
    # Run operator with disabled warning
    with np.errstate(invalid="ignore", divide="ignore"):
        return operator_fn(b, a)


@register_numpy_types(numeric.finfo)
def _(a: np.ndarray) -> TypeInfo:
    ti = np.finfo(a.dtype)
    return TypeInfo(ti.eps, ti.max, ti.min)


@register_numpy_types(numeric.clip)
def _(
    a: Union[np.ndarray, np.generic],
    a_min: Union[np.ndarray, np.generic, float],
    a_max: Union[np.ndarray, np.generic, float],
) -> Union[np.ndarray, np.generic]:
    return np.clip(a, a_min, a_max)


@register_numpy_types(numeric.as_tensor_like)
def _(a: Union[np.ndarray, np.generic], data: Any) -> Union[np.ndarray, np.generic]:
    return np.array(data)


@register_numpy_types(numeric.item)
def _(a: Union[np.ndarray, np.generic]) -> Union[int, float, bool]:
    return a.item()


@register_numpy_types(numeric.sum)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.array(np.sum(a, axis=axis, keepdims=keepdims))


@register_numpy_types(numeric.multiply)
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.multiply(x1, x2)


@register_numpy_types(numeric.var)
def _(
    a: Union[np.ndarray, np.generic],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> np.ndarray:
    return np.array(np.var(a, axis=axis, keepdims=keepdims, ddof=ddof))


@register_numpy_types(numeric.size)
def _(a: Union[np.ndarray, np.generic]) -> int:
    return a.size


@register_numpy_types(numeric.matmul)
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.matmul(x1, x2)


@register_numpy_types(numeric.unsqueeze)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[np.ndarray, np.generic]:
    return np.expand_dims(a, axis=axis)


@register_numpy_types(numeric.transpose)
def _(a: Union[np.ndarray, np.generic], axes: Optional[Tuple[int, ...]] = None) -> Union[np.ndarray, np.generic]:
    return np.transpose(a, axes=axes)


@register_numpy_types(numeric.argsort)
def _(
    a: Union[np.ndarray, np.generic], axis: int = -1, descending=False, stable=False
) -> Union[np.ndarray, np.generic]:
    if descending and stable:
        return a.shape[axis] - 1 - np.flip(np.argsort(np.flip(a, axis), axis=axis, kind="stable"), axis)
    if descending and not stable:
        return np.flip(np.argsort(a, axis=axis), axis)
    return np.argsort(a, axis=axis, kind="stable" if stable else None)


@register_numpy_types(numeric.diag)
def _(a: Union[np.ndarray, np.generic], k: int = 0) -> np.ndarray:
    return np.diag(a, k=k)


@register_numpy_types(numeric.logical_or)
def _(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.logical_or(x1, x2)


@register_numpy_types(numeric.masked_mean)
def _(
    x: np.ndarray, mask: Optional[np.ndarray], axis: Union[int, Tuple[int, ...], List[int]], keepdims=False
) -> np.ndarray:
    if mask is None:
        return np.mean(x, axis=axis, keepdims=keepdims)
    masked_x = np.ma.array(x, mask=mask)
    result = np.ma.mean(masked_x, axis=axis, keepdims=keepdims)
    if isinstance(result, np.ma.MaskedArray):
        return result.data
    return result


@register_numpy_types(numeric.masked_median)
def _(
    x: np.ndarray, mask: Optional[np.ndarray], axis: Union[int, Tuple[int, ...], List[int]], keepdims=False
) -> np.ndarray:
    if mask is None:
        return np.median(x, axis=axis, keepdims=keepdims)
    masked_x = np.ma.array(x, mask=mask)
    result = np.ma.median(masked_x, axis=axis, keepdims=keepdims)
    if isinstance(result, np.ma.MaskedArray):
        return result.data
    return result


@register_numpy_types(numeric.expand_dims)
def _(a: np.ndarray, axis: Union[int, Tuple[int, ...], List[int]]) -> np.ndarray:
    return np.expand_dims(a, axis=axis)


@register_numpy_types(numeric.clone)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return a.copy()


@register_numpy_types(numeric.searchsorted)
def _(a: np.ndarray, v: np.ndarray, side: str = "left", sorter: Optional[np.ndarray] = None) -> np.ndarray:
    return np.searchsorted(a, v, side, sorter)


def zeros(
    shape: Tuple[int, ...],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> np.ndarray:
    validate_device(device)
    dtype = convert_to_numpy_dtype(dtype)
    return np.zeros(shape, dtype=dtype)


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> np.ndarray:
    validate_device(device)
    dtype = convert_to_numpy_dtype(dtype)
    return np.eye(n, m, dtype=dtype)


def arange(
    start: float,
    end: float,
    step: float,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> np.ndarray:
    validate_device(device)
    dtype = convert_to_numpy_dtype(dtype)
    return np.arange(start, end, step, dtype=dtype)


@register_numpy_types(numeric.log2)
def _(a: Union[np.ndarray, np.generic]) -> Union[np.ndarray, np.generic]:
    return np.log2(a)


@register_numpy_types(numeric.ceil)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.ceil(a)


def tensor(
    data: Union[TTensor, Sequence[float]],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> np.ndarray:
    validate_device(device)
    dtype = convert_to_numpy_dtype(dtype)
    return np.array(data, dtype=dtype)
