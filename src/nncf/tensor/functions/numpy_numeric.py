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

from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import DTypeLike
from numpy.typing import NDArray

from nncf.tensor.definitions import T_AXIS
from nncf.tensor.definitions import T_NUMBER
from nncf.tensor.definitions import T_SHAPE
from nncf.tensor.definitions import T_SHAPE_ARRAY
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric as numeric
from nncf.tensor.tensor import TTensor

T_NUMPY_ARRAY = NDArray[Any]
T_NUMPY = Union[T_NUMPY_ARRAY, np.generic]  # type: ignore[type-arg]

DTYPE_MAP: dict[TensorDataType, DTypeLike] = {
    TensorDataType.float16: np.dtype(np.float16),
    TensorDataType.float32: np.dtype(np.float32),
    TensorDataType.float64: np.dtype(np.float64),
    TensorDataType.int8: np.dtype(np.int8),
    TensorDataType.int32: np.dtype(np.int32),
    TensorDataType.int64: np.dtype(np.int64),
    TensorDataType.uint8: np.dtype(np.uint8),
    TensorDataType.uint16: np.dtype(np.uint16),
    TensorDataType.uint32: np.dtype(np.uint32),
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


def validate_device(device: Optional[TensorDeviceType]) -> None:
    if device is not None and device != TensorDeviceType.CPU:
        msg = "numpy_numeric only supports CPU device."
        raise ValueError(msg)


def convert_to_numpy_dtype(dtype: Optional[TensorDataType]) -> Optional[DTypeLike]:
    return DTYPE_MAP[dtype] if dtype is not None else None


@numeric.device.register
def _(a: T_NUMPY) -> TensorDeviceType:
    return TensorDeviceType.CPU


@numeric.backend.register
def _(a: T_NUMPY) -> TensorBackend:
    return TensorBackend.numpy


@numeric.bincount.register
def _(a: T_NUMPY, *, weights: Optional[T_NUMPY], minlength: int = 0) -> T_NUMPY:
    return np.bincount(a, weights=weights, minlength=minlength)


@numeric.squeeze.register
def _(a: T_NUMPY, axis: T_AXIS = None) -> T_NUMPY:
    return np.squeeze(a, axis=axis)


@numeric.flatten.register
def _(a: T_NUMPY) -> T_NUMPY_ARRAY:
    return a.flatten()


@numeric.max.register
def _(a: T_NUMPY, axis: T_AXIS = None, keepdims: bool = False) -> T_NUMPY_ARRAY:
    return np.array(np.max(a, axis=axis, keepdims=keepdims))


@numeric.min.register
def _(a: T_NUMPY, axis: T_AXIS = None, keepdims: bool = False) -> T_NUMPY:
    return np.array(np.min(a, axis=axis, keepdims=keepdims))


@numeric.abs.register
def _(a: T_NUMPY) -> T_NUMPY:
    return np.absolute(a)


@numeric.astype.register
def _(a: T_NUMPY, dtype: TensorDataType) -> T_NUMPY:
    return a.astype(DTYPE_MAP[dtype])


@numeric.view.register
def _(a: T_NUMPY, dtype: TensorDataType) -> T_NUMPY:
    return a.view(DTYPE_MAP[dtype])


@numeric.dtype.register
def _(a: T_NUMPY) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@numeric.repeat.register
def _(a: T_NUMPY, repeats: Union[int, T_NUMPY_ARRAY], *, axis: Optional[int] = None) -> T_NUMPY:
    return np.repeat(a, repeats=repeats, axis=axis)


@numeric.reshape.register
def _(a: T_NUMPY, shape: T_SHAPE) -> T_NUMPY:
    return a.reshape(shape)


@numeric.atleast_1d.register
def _(a: T_NUMPY_ARRAY) -> T_NUMPY_ARRAY:
    return np.atleast_1d(a)


@numeric.all.register
def _(a: T_NUMPY, axis: T_AXIS = None) -> T_NUMPY_ARRAY:
    return np.array(np.all(a, axis=axis))


@numeric.allclose.register
def _(
    a: T_NUMPY,
    b: Union[T_NUMPY, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.any.register
def _(a: T_NUMPY, axis: T_AXIS = None) -> T_NUMPY_ARRAY:
    return np.array(np.any(a, axis=axis))


@numeric.count_nonzero.register
def _(a: T_NUMPY, axis: T_AXIS = None) -> T_NUMPY_ARRAY:
    return np.array(np.count_nonzero(a, axis=axis))


@numeric.histogram.register
def _(
    a: T_NUMPY,
    bins: int,
    *,
    range: Optional[tuple[float, float]] = None,
) -> T_NUMPY:
    return np.histogram(a=a, bins=bins, range=range)[0]


@numeric.isempty.register
def _(a: T_NUMPY) -> bool:
    return a.size == 0


@numeric.isclose.register
def _(
    a: T_NUMPY,
    b: Union[T_NUMPY, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> T_NUMPY_ARRAY:
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.maximum.register
def _(x1: T_NUMPY, x2: Union[T_NUMPY, float]) -> T_NUMPY_ARRAY:
    return np.maximum(x1, x2)


@numeric.minimum.register
def _(x1: T_NUMPY, x2: Union[T_NUMPY, float]) -> T_NUMPY_ARRAY:
    return np.minimum(x1, x2)


@numeric.ones_like.register
def _(a: T_NUMPY) -> T_NUMPY_ARRAY:
    return np.ones_like(a)


@numeric.where.register
def _(
    condition: T_NUMPY,
    x: Union[T_NUMPY, float],
    y: Union[T_NUMPY, float],
) -> T_NUMPY_ARRAY:
    return np.where(condition, x, y)


@numeric.zeros_like.register
def _(a: T_NUMPY) -> T_NUMPY_ARRAY:
    return np.zeros_like(a)


@numeric.stack.register
def _(x: Sequence[T_NUMPY], axis: int = 0) -> T_NUMPY_ARRAY:
    return np.stack(x, axis=axis)


@numeric.concatenate.register
def _(x: T_NUMPY, axis: int = 0) -> T_NUMPY_ARRAY:
    return np.concatenate(x, axis=axis)


@numeric.unstack.register
def _(x: T_NUMPY, axis: int = 0) -> list[T_NUMPY_ARRAY]:
    return [np.squeeze(e, axis) for e in np.split(x, x.shape[axis], axis=axis)]


@numeric.moveaxis.register
def _(a: T_NUMPY_ARRAY, source: Union[int, tuple[int, ...]], destination: Union[int, tuple[int, ...]]) -> T_NUMPY_ARRAY:
    return np.moveaxis(a, source, destination)


@numeric.mean.register
def _(
    a: T_NUMPY,
    axis: T_AXIS = None,
    keepdims: bool = False,
    dtype: Optional[TensorDataType] = None,
) -> T_NUMPY_ARRAY:
    np_dtype = convert_to_numpy_dtype(dtype)
    return np.array(np.mean(a, axis=axis, keepdims=keepdims, dtype=np_dtype))  # type: ignore [arg-type]


@numeric.median.register
def _(
    a: T_NUMPY,
    axis: Optional[T_SHAPE] = None,
    keepdims: bool = False,
) -> T_NUMPY_ARRAY:
    return np.array(np.median(a, axis=axis, keepdims=keepdims))  # type: ignore [arg-type]


@numeric.floor.register
def _(a: T_NUMPY) -> T_NUMPY:
    return np.floor(a)


@numeric.round.register
def _(a: T_NUMPY, decimals: int = 0) -> T_NUMPY_ARRAY:
    return np.round(a, decimals=decimals)


@numeric.power.register
def _(a: T_NUMPY, exponent: Union[T_NUMPY, float]) -> T_NUMPY:
    return np.power(a, exponent)


@numeric.quantile.register
def _(
    a: T_NUMPY,
    q: Union[float, list[float]],
    axis: T_AXIS = None,
    keepdims: bool = False,
) -> T_NUMPY:
    return np.array(np.quantile(a.astype(np.float64, copy=False), q=q, axis=axis, keepdims=keepdims))


@numeric.percentile.register
def _(
    a: T_NUMPY,
    q: Union[float, list[float]],
    axis: T_AXIS,
    keepdims: bool = False,
) -> T_NUMPY:
    return np.quantile(
        a.astype(np.float64, copy=False), q=np.true_divide(np.array(q), 100), axis=axis, keepdims=keepdims
    )


@numeric._binary_op_nowarn.register
def _(a: T_NUMPY, b: Union[T_NUMPY, float], operator_fn: Callable[..., Any]) -> T_NUMPY:
    # Run operator with disabled warning
    with np.errstate(invalid="ignore", divide="ignore"):
        return operator_fn(a, b)


@numeric._binary_reverse_op_nowarn.register
def _(a: T_NUMPY, b: Union[T_NUMPY, float], operator_fn: Callable[..., Any]) -> T_NUMPY:
    # Run operator with disabled warning
    with np.errstate(invalid="ignore", divide="ignore"):
        return operator_fn(b, a)


@numeric.finfo.register
def _(a: T_NUMPY) -> TypeInfo:
    ti = np.finfo(a.dtype)  # type: ignore[arg-type]
    return TypeInfo(float(ti.eps), float(ti.max), float(ti.min))


@numeric.clip.register
def _(
    a: T_NUMPY,
    a_min: Union[T_NUMPY, float],
    a_max: Union[T_NUMPY, float],
) -> T_NUMPY:
    return np.clip(a, a_min, a_max)


@numeric.as_tensor_like.register
def _(a: T_NUMPY, data: Any) -> T_NUMPY:
    return np.array(data)


@numeric.item.register
def _(a: T_NUMPY) -> T_NUMBER:
    return a.item()


@numeric.cumsum.register
def _(a: T_NUMPY, axis: int) -> T_NUMPY:
    return np.cumsum(a, axis=axis)


@numeric.sum.register
def _(a: T_NUMPY, axis: T_AXIS = None, keepdims: bool = False) -> T_NUMPY_ARRAY:
    return np.array(np.sum(a, axis=axis, keepdims=keepdims))


@numeric.multiply.register
def _(x1: T_NUMPY, x2: Union[T_NUMPY, float]) -> T_NUMPY_ARRAY:
    return np.multiply(x1, x2)


@numeric.var.register
def _(
    a: T_NUMPY,
    axis: T_AXIS = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> T_NUMPY_ARRAY:
    return np.array(np.var(a, axis=axis, keepdims=keepdims, ddof=ddof))  # type: ignore[arg-type]


@numeric.size.register
def _(a: T_NUMPY) -> int:
    return a.size


@numeric.matmul.register
def _(x1: T_NUMPY, x2: Union[T_NUMPY, float]) -> T_NUMPY_ARRAY:
    return np.matmul(x1, x2)


@numeric.unsqueeze.register
def _(a: T_NUMPY, axis: int) -> T_NUMPY:
    return np.expand_dims(a, axis=axis)


@numeric.transpose.register
def _(a: T_NUMPY, axes: Optional[T_SHAPE_ARRAY] = None) -> T_NUMPY:
    return np.transpose(a, axes=axes)


@numeric.argsort.register
def _(a: T_NUMPY, axis: int = -1, descending: bool = False, stable: bool = False) -> T_NUMPY:
    if descending and stable:
        return a.shape[axis] - 1 - np.flip(np.argsort(np.flip(a, axis), axis=axis, kind="stable"), axis)
    if descending and not stable:
        return np.flip(np.argsort(a, axis=axis), axis)
    return np.argsort(a, axis=axis, kind="stable" if stable else None)


@numeric.diag.register
def _(a: T_NUMPY, k: int = 0) -> T_NUMPY_ARRAY:
    return np.diag(a, k=k)


@numeric.logical_or.register
def _(x1: T_NUMPY_ARRAY, x2: T_NUMPY_ARRAY) -> T_NUMPY_ARRAY:
    return np.logical_or(x1, x2)


@numeric.masked_mean.register
def _(
    x: T_NUMPY_ARRAY,
    mask: Optional[T_NUMPY_ARRAY],
    axis: T_AXIS,
    keepdims: bool = False,
) -> T_NUMPY_ARRAY:
    if mask is None:
        return np.mean(x, axis=axis, keepdims=keepdims)
    masked_x = np.ma.array(x, mask=mask)  # type: ignore[no-untyped-call]
    result = np.ma.mean(masked_x, axis=axis, keepdims=keepdims)
    if isinstance(result, np.ma.MaskedArray):
        return result.data
    return result


@numeric.masked_median.register
def _(x: T_NUMPY_ARRAY, mask: Optional[T_NUMPY_ARRAY], axis: T_AXIS, keepdims: bool = False) -> T_NUMPY_ARRAY:
    if mask is None:
        return np.median(x, axis=axis, keepdims=keepdims)
    masked_x = np.ma.array(x, mask=mask)  # type: ignore[no-untyped-call]
    result = np.ma.median(masked_x, axis=axis, keepdims=keepdims)  # type: ignore[no-untyped-call]
    if isinstance(result, np.ma.MaskedArray):
        return result.data
    return result


@numeric.expand_dims.register
def _(a: T_NUMPY, axis: T_SHAPE) -> T_NUMPY_ARRAY:
    return np.expand_dims(a, axis=axis)


@numeric.clone.register
def _(a: T_NUMPY) -> T_NUMPY:
    return a.copy()


@numeric.searchsorted.register
def _(
    a: T_NUMPY_ARRAY, v: T_NUMPY_ARRAY, side: Literal["left", "right"] = "left", sorter: Optional[T_NUMPY_ARRAY] = None
) -> T_NUMPY_ARRAY:
    return np.searchsorted(a, v, side, sorter)


@numeric.as_numpy_tensor.register
def _(a: T_NUMPY_ARRAY) -> T_NUMPY_ARRAY:
    return a


def zeros(
    shape: tuple[int, ...],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> T_NUMPY_ARRAY:
    validate_device(device)
    np_dtype = convert_to_numpy_dtype(dtype)
    return np.zeros(shape, dtype=np_dtype)


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> T_NUMPY_ARRAY:
    validate_device(device)
    np_dtype = convert_to_numpy_dtype(dtype)
    return np.eye(n, m, dtype=np_dtype)


def linspace(
    start: float,
    end: float,
    num: int,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> T_NUMPY_ARRAY:
    validate_device(device)
    np_dtype = convert_to_numpy_dtype(dtype)
    return np.linspace(start, end, num, dtype=np_dtype)


def arange(
    start: float,
    end: float,
    step: float,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> T_NUMPY_ARRAY:
    validate_device(device)
    np_dtype = convert_to_numpy_dtype(dtype)
    return np.arange(start, end, step, dtype=np_dtype)


@numeric.log2.register
def _(a: T_NUMPY) -> T_NUMPY:
    return np.log2(a)


@numeric.ceil.register
def _(a: T_NUMPY) -> T_NUMPY_ARRAY:
    return np.ceil(a)


def tensor(
    data: Union[TTensor, list[float]],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> T_NUMPY_ARRAY:
    validate_device(device)
    np_dtype = convert_to_numpy_dtype(dtype)
    return np.array(data, dtype=np_dtype)


@numeric.tolist.register
def _(a: T_NUMPY) -> Any:
    return a.tolist()
