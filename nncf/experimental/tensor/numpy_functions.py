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

from typing import List, Optional, Tuple, Union

import numpy as np

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
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> np.ndarray:
    return np.max(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.min)
def _(
    a: Union[np.ndarray, np.generic], axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False
) -> Union[np.ndarray, np.generic]:
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
def _(a: np.ndarray, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> np.ndarray:
    return np.moveaxis(a, source, destination)


@_register_numpy_types(fns.mean)
def _(a: Union[np.ndarray, np.generic], axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray:
    return np.mean(a, axis=axis, keepdims=keepdims)


@_register_numpy_types(fns.round)
def _(a: Union[np.ndarray, np.generic], decimals: int = 0) -> np.ndarray:
    return np.round(a, decimals=decimals)
