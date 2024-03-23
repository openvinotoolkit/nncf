# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np

from nncf.experimental.tensor.definitions import TensorDataType
from nncf.experimental.tensor.definitions import TensorDeviceType
from nncf.experimental.tensor.definitions import TypeInfo
from nncf.experimental.tensor.functions import numeric as numeric
from nncf.experimental.tensor.functions.dispatcher import register_numpy_types

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


@register_numpy_types(numeric.device)
def _(a: Union[np.ndarray, np.generic]) -> TensorDeviceType:
    return TensorDeviceType.CPU


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
    dtype = DTYPE_MAP[dtype] if dtype else None
    return np.array(np.mean(a, axis=axis, keepdims=keepdims, dtype=dtype))


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
    keepdims: Optional[bool] = None,
) -> Union[np.ndarray, np.generic]:
    return np.array(np.quantile(a, q=q, axis=axis, keepdims=keepdims))


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
    a: Union[np.ndarray, np.generic], axis: Optional[int] = None, descending=False, stable=False
) -> Union[np.ndarray, np.generic]:
    return np.argsort(a, axis=axis)
