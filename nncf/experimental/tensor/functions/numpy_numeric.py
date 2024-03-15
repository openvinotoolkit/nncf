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

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from nncf.experimental.tensor.definitions import TensorDataType
from nncf.experimental.tensor.definitions import TensorDeviceType
from nncf.experimental.tensor.definitions import TypeInfo
from nncf.experimental.tensor.functions import numeric as numeric

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


@numeric.device.register
def _(a: Union[np.ndarray, np.generic]) -> TensorDeviceType:
    return TensorDeviceType.CPU


@numeric.squeeze.register
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[np.ndarray, np.generic]:
    return np.squeeze(a, axis=axis)


@numeric.flatten.register
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return a.flatten()


@numeric.max.register
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.array(np.max(a, axis=axis, keepdims=keepdims))


@numeric.min.register
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> Union[np.ndarray, np.generic]:
    return np.array(np.min(a, axis=axis, keepdims=keepdims))


@numeric.abs.register
def _(a: Union[np.ndarray, np.generic]) -> Union[np.ndarray, np.generic]:
    return np.absolute(a)


@numeric.astype.register
def _(a: Union[np.ndarray, np.generic], dtype: TensorDataType) -> Union[np.ndarray, np.generic]:
    return a.astype(DTYPE_MAP[dtype])


@numeric.dtype.register
def _(a: Union[np.ndarray, np.generic]) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@numeric.reshape.register
def _(a: Union[np.ndarray, np.generic], shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return a.reshape(shape)


@numeric.all.register
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.all(a, axis=axis))


@numeric.allclose.register
def _(
    a: Union[np.ndarray, np.generic],
    b: Union[np.ndarray, np.generic, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.any.register
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.any(a, axis=axis))


@numeric.count_nonzero.register
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.count_nonzero(a, axis=axis))


@numeric.isempty.register
def _(a: Union[np.ndarray, np.generic]) -> bool:
    return a.size == 0


@numeric.isclose.register
def _(
    a: Union[np.ndarray, np.generic],
    b: Union[np.ndarray, np.generic, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> np.ndarray:
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.maximum.register
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.maximum(x1, x2)


@numeric.minimum.register
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.minimum(x1, x2)


@numeric.ones_like.register
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.ones_like(a)


@numeric.where.register
def _(
    condition: Union[np.ndarray, np.generic],
    x: Union[np.ndarray, np.generic, float],
    y: Union[np.ndarray, np.generic, float],
) -> np.ndarray:
    return np.where(condition, x, y)


@numeric.zeros_like.register
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.zeros_like(a)


@numeric.stack.register
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return np.stack(x, axis=axis)


@numeric.unstack.register
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return [np.squeeze(e, axis) for e in np.split(x, x.shape[axis], axis=axis)]


@numeric.moveaxis.register
def _(a: np.ndarray, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return np.moveaxis(a, source, destination)


@numeric.mean.register
def _(a: Union[np.ndarray, np.generic], axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray:
    return np.array(np.mean(a, axis=axis, keepdims=keepdims))


@numeric.round.register
def _(a: Union[np.ndarray, np.generic], decimals: int = 0) -> np.ndarray:
    return np.round(a, decimals=decimals)


@numeric.power.register
def _(a: Union[np.ndarray, np.generic], exponent: Union[np.ndarray, float]) -> Union[np.ndarray, np.generic]:
    return np.power(a, exponent)


@numeric.quantile.register
def _(
    a: Union[np.ndarray, np.generic],
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = None,
) -> Union[np.ndarray, np.generic]:
    return np.array(np.quantile(a, q=q, axis=axis, keepdims=keepdims))


@numeric.finfo.register
def _(a: Union[np.ndarray, np.generic]) -> TypeInfo:
    ti = np.finfo(a.dtype)
    return TypeInfo(ti.eps, ti.max, ti.min)


@numeric.clip.register
def _(
    a: Union[np.ndarray, np.generic],
    a_min: Union[np.ndarray, np.generic, float],
    a_max: Union[np.ndarray, np.generic, float],
) -> Union[np.ndarray, np.generic]:
    return np.clip(a, a_min, a_max)


@numeric.as_tensor_like.register
def _(a: Union[np.ndarray, np.generic], data: Any) -> Union[np.ndarray, np.generic]:
    return np.array(data)


@numeric.item.register
def _(a: Union[np.ndarray, np.generic]) -> Union[int, float, bool]:
    return a.item()


@numeric.sum.register
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.array(np.sum(a, axis=axis, keepdims=keepdims))


@numeric.multiply.register
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.multiply(x1, x2)


@numeric.var.register
def _(
    a: Union[np.ndarray, np.generic],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: bool = False,
    ddof: int = 0,
) -> np.ndarray:
    return np.array(np.var(a, axis=axis, keepdims=keepdims, ddof=ddof))


@numeric.size.register
def _(a: Union[np.ndarray, np.generic]) -> int:
    return a.size


@numeric.matmul.register
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.matmul(x1, x2)


@numeric.unsqueeze.register
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[np.ndarray, np.generic]:
    return np.expand_dims(a, axis=axis)


@numeric.transpose.register
def _(a: Union[np.ndarray, np.generic], axes: Optional[Tuple[int, ...]] = None) -> Union[np.ndarray, np.generic]:
    return np.transpose(a, axes=axes)


@numeric.argsort.register
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[int] = None, descending=False, stable=False
) -> Union[np.ndarray, np.generic]:
    return np.argsort(a, axis=axis)
