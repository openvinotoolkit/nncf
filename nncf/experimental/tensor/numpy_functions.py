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
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant

from nncf.experimental.tensor import functions as fns
from nncf.experimental.tensor.enums import TensorDataType
from nncf.experimental.tensor.enums import TensorDeviceType

DTYPE_MAP = {
    TensorDataType.float16: np.dtype(np.float16),
    TensorDataType.float32: np.dtype(np.float32),
    TensorDataType.float64: np.dtype(np.float64),
    TensorDataType.int8: np.dtype(np.int8),
    TensorDataType.uint8: np.dtype(np.uint8),
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


def _register_numpy_types(singledispatch_fn):
    """
    Decorator to register function to singledispatch for numpy classes.

    :param singledispatch_fn: singledispatch function.
    """

    def inner(func):
        singledispatch_fn.register(np.ndarray)(func)
        singledispatch_fn.register(np.generic)(func)
        singledispatch_fn.register(float)(func)  # np.min and np.max with keepdims=False and no axes return `float`
        return func

    return inner


@_register_numpy_types(fns.device)
def _(a: Union[np.ndarray, np.generic]) -> TensorDeviceType:
    return TensorDeviceType.CPU


@_register_numpy_types(fns.squeeze)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[np.ndarray, np.generic]:
    return np.squeeze(a, axis=axis)


@_register_numpy_types(fns.flatten)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return a.flatten()


@_register_numpy_types(fns.max)
def _(
    a: Union[np.ndarray, np.generic],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = None,
) -> np.ndarray:
    if keepdims is None:
        keepdims = np._NoValue
    return np.max(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.min)
def _(
    a: Union[np.ndarray, np.generic],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdims: Optional[bool] = None,
) -> Union[np.ndarray, np.generic]:
    if keepdims is None:
        keepdims = np._NoValue
    return np.min(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.abs)
def _(a: Union[np.ndarray, np.generic]) -> Union[np.ndarray, np.generic]:
    return np.absolute(a)


@_register_numpy_types(fns.astype)
def _(a: Union[np.ndarray, np.generic], dtype: TensorDataType) -> Union[np.ndarray, np.generic]:
    return a.astype(DTYPE_MAP[dtype])


@_register_numpy_types(fns.dtype)
def _(a: Union[np.ndarray, np.generic]) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@_register_numpy_types(fns.reshape)
def _(a: Union[np.ndarray, np.generic], shape: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return a.reshape(shape)


@_register_numpy_types(fns.all)
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[np.ndarray, bool]:
    return np.all(a, axis=axis)


@_register_numpy_types(fns.allclose)
def _(
    a: Union[np.ndarray, np.generic],
    b: Union[np.ndarray, np.generic, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> np.ndarray:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@_register_numpy_types(fns.any)
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[np.ndarray, bool]:
    return np.any(a, axis=axis)


@_register_numpy_types(fns.count_nonzero)
def _(a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None) -> np.ndarray:
    return np.array(np.count_nonzero(a, axis=axis))


@_register_numpy_types(fns.isempty)
def _(a: Union[np.ndarray, np.generic]) -> bool:
    return a.size == 0


@_register_numpy_types(fns.isclose)
def _(
    a: Union[np.ndarray, np.generic],
    b: Union[np.ndarray, np.generic, float],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> Union[np.ndarray, bool]:
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@_register_numpy_types(fns.maximum)
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.maximum(x1, x2)


@_register_numpy_types(fns.minimum)
def _(x1: Union[np.ndarray, np.generic], x2: Union[np.ndarray, np.generic, float]) -> np.ndarray:
    return np.minimum(x1, x2)


@_register_numpy_types(fns.ones_like)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.ones_like(a)


@_register_numpy_types(fns.where)
def _(
    condition: Union[np.ndarray, np.generic],
    x: Union[np.ndarray, np.generic, float],
    y: Union[np.ndarray, np.generic, float],
) -> np.ndarray:
    return np.where(condition, x, y)


@_register_numpy_types(fns.zeros_like)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return np.zeros_like(a)


@_register_numpy_types(fns.stack)
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return np.stack(x, axis=axis)


@_register_numpy_types(fns.unstack)
def _(x: Union[np.ndarray, np.generic], axis: int = 0) -> List[np.ndarray]:
    return [np.squeeze(e, axis) for e in np.split(x, x.shape[axis], axis=axis)]


@_register_numpy_types(fns.moveaxis)
def _(a: np.ndarray, source: Union[int, List[int]], destination: Union[int, List[int]]) -> np.ndarray:
    return np.moveaxis(a, source, destination)


@_register_numpy_types(fns.mean)
def _(
    a: Union[np.ndarray, np.generic], axis: Union[int, Tuple[int, ...]] = None, keepdims: Optional[bool] = None
) -> np.ndarray:
    if keepdims is None:
        keepdims = np._NoValue
    return np.mean(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.round)
def _(a: Union[np.ndarray, np.generic], decimals: int = 0) -> np.ndarray:
    return np.round(a, decimals=decimals)


@_register_numpy_types(fns.binary_operator)
def _(
    a: Union[np.ndarray, np.generic], b: Union[np.ndarray, np.generic], operator_fn: Callable
) -> Union[np.ndarray, np.generic]:
    # Run operator with disabled warning
    with np.errstate(invalid="ignore", divide="ignore"):
        return operator_fn(a, b)


@_register_numpy_types(fns.binary_reverse_operator)
def _(
    a: Union[np.ndarray, np.generic], b: Union[np.ndarray, np.generic], operator_fn: Callable
) -> Union[np.ndarray, np.generic]:
    # Run operator with disabled warning
    with np.errstate(invalid="ignore", divide="ignore"):
        return operator_fn(b, a)


@_register_numpy_types(fns.to_numpy)
def _(a: Union[np.ndarray, np.generic]) -> np.ndarray:
    return a


@_register_numpy_types(fns.inf)
def _(a: Union[np.ndarray, np.generic]) -> Any:
    return np.inf


@_register_numpy_types(fns.concatenate)
def _(x: List[np.ndarray], axis: int = 0) -> np.ndarray:
    return np.concatenate(x, axis=axis)


@_register_numpy_types(fns.min_of_list)
def _(x: List[np.ndarray], axis: int = 0) -> np.ndarray:
    return np.min(x, axis=axis)


@_register_numpy_types(fns.max_of_list)
def _(x: List[np.ndarray], axis: int = 0) -> np.ndarray:
    return np.max(x, axis=axis)


@_register_numpy_types(fns.amax)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[List[int]] = None, keepdims: Optional[bool] = None
) -> Union[np.ndarray, np.generic]:
    if keepdims is None:
        keepdims = np._NoValue
    return np.amax(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.amin)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[List[int]] = None, keepdims: Optional[bool] = None
) -> Union[np.ndarray, np.generic]:
    if keepdims is None:
        keepdims = np._NoValue
    return np.amin(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.clip)
def _(
    a: Union[np.ndarray, np.generic], min_val: float, max_val: Optional[float] = None
) -> Union[np.ndarray, np.generic]:
    return np.clip(a, a_min=min_val, a_max=max_val)


@_register_numpy_types(fns.sum)
def _(a: Union[np.ndarray, np.generic], axes: List[int]) -> Union[np.ndarray, np.generic]:
    return np.sum(a, axis=tuple(axes))


@_register_numpy_types(fns.transpose)
def _(a: Union[np.ndarray, np.generic], axes: List[int]) -> Union[np.ndarray, np.generic]:
    return np.transpose(a, axes=axes)


@_register_numpy_types(fns.eps)
def _(a: Union[np.ndarray, np.generic], dtype: TensorDataType) -> float:
    return np.finfo(DTYPE_MAP[dtype]).eps


@_register_numpy_types(fns.median)
def _(
    a: Union[np.ndarray, np.generic], axis: Union[int, Tuple[int]] = None, keepdims: Optional[bool] = None
) -> Union[np.ndarray, np.generic]:
    if keepdims is None:
        keepdims = np._NoValue
    return np.median(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.power)
def _(a: Union[np.ndarray, np.generic], pwr: float) -> Union[np.ndarray, np.generic]:
    return np.power(a, pwr)


@_register_numpy_types(fns.quantile)
def _(
    a: Union[np.ndarray, np.generic],
    q: Union[float, List[float]],
    axis: Union[int, List[int]] = None,
    keepdims: Optional[bool] = None,
) -> Union[float, Union[np.ndarray, np.generic]]:
    if keepdims is None:
        keepdims = np._NoValue
    return np.quantile(a, q=q, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.matmul)
def _(a: Union[np.ndarray, np.generic], b: Union[np.ndarray, np.generic]) -> Union[np.ndarray, np.generic]:
    return np.matmul(a, b)


@_register_numpy_types(fns.logical_or)
def _(tensor1: Union[np.ndarray, np.generic], tensor2: Union[np.ndarray, np.generic]) -> Union[np.ndarray, np.generic]:
    return np.logical_or(tensor1, tensor2)


@_register_numpy_types(fns.masked_mean)
def _(
    a: Union[np.ndarray, np.generic],
    mask: Union[np.ndarray, np.generic],
    axis: int = None,
    keepdims: Optional[bool] = None,
) -> Union[np.ndarray, np.generic]:
    if keepdims is None:
        keepdims = np._NoValue
    masked_x = np.ma.array(a, mask=mask)
    result = np.ma.mean(masked_x, axis=axis, keepdims=keepdims)
    if isinstance(result, (MaskedConstant, MaskedArray)):
        result = result.data
    return result


@_register_numpy_types(fns.masked_median)
def _(
    a: Union[np.ndarray, np.generic],
    mask: Union[np.ndarray, np.generic],
    axis: int = None,
    keepdims: Optional[bool] = None,
) -> Union[np.ndarray, np.generic]:
    if keepdims is None:
        keepdims = np._NoValue
    masked_x = np.ma.array(a, mask=mask)
    result = np.ma.median(masked_x, axis=axis, keepdims=keepdims)
    if isinstance(result, (MaskedConstant, MaskedArray)):
        result = result.data
    return result


@_register_numpy_types(fns.size)
def _(a: Union[np.ndarray, np.generic]) -> int:
    return a.size
