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

from nncf.experimental.tensor import functions
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


def registry_numpy_types(registry_fn):
    def inner(func):
        registry_fn.register(np.ndarray)(func)
        registry_fn.register(np.generic)(func)
        return func

    return inner


@registry_numpy_types(functions.device)
def _(a: Union[np.ndarray, np.number]) -> TensorDeviceType:
    return TensorDeviceType.CPU


@registry_numpy_types(functions.squeeze)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.squeeze(a, axis=axis)


@registry_numpy_types(functions.flatten)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return a.flatten()


@registry_numpy_types(functions.max)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.max(a, axis=axis)


@registry_numpy_types(functions.amax)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.amax(a, axis=axis)


@registry_numpy_types(functions.min)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.min(a, axis=axis)


@registry_numpy_types(functions.amin)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.amin(a, axis=axis)


@registry_numpy_types(functions.abs)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return np.absolute(a)


@registry_numpy_types(functions.astype)
def _(a: Union[np.ndarray, np.number], dtype: TensorDataType) -> np.ndarray:
    return a.astype(DTYPE_MAP[dtype])


@registry_numpy_types(functions.dtype)
def _(a: Union[np.ndarray, np.number]) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@registry_numpy_types(functions.reshape)
def _(a: Union[np.ndarray, np.number], shape: Union[int, Tuple[int]]) -> np.ndarray:
    return a.reshape(shape)


@registry_numpy_types(functions.all)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> Union[np.ndarray, bool]:
    return np.all(a, axis=axis)


@registry_numpy_types(functions.allclose)
def _(
    a: Union[np.ndarray, np.number],
    b: Union[np.ndarray, np.number],
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@registry_numpy_types(functions.any)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> Union[np.ndarray, bool]:
    return np.any(a, axis=axis)


@registry_numpy_types(functions.count_nonzero)
def _(a: Union[np.ndarray, np.number], axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.count_nonzero(a, axis=axis)


@registry_numpy_types(functions.isempty)
def _(a: Union[np.ndarray, np.number]) -> bool:
    return a.size == 0


@registry_numpy_types(functions.isclose)
def _(
    a: Union[np.ndarray, np.number],
    b: np.ndarray,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    equal_nan: bool = False,
):
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@registry_numpy_types(functions.maximum)
def _(x1: Union[np.ndarray, np.number], x2: np.ndarray) -> np.ndarray:
    return np.maximum(x1, x2)


@registry_numpy_types(functions.minimum)
def _(x1: Union[np.ndarray, np.number], x2: np.ndarray) -> np.ndarray:
    return np.minimum(x1, x2)


@registry_numpy_types(functions.ones_like)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return np.ones_like(a)


@registry_numpy_types(functions.where)
def _(
    condition: Union[np.ndarray, np.number],
    x: Union[np.ndarray, np.number, float, bool],
    y: Union[np.ndarray, float, bool],
) -> np.ndarray:
    return np.where(condition, x, y)


@registry_numpy_types(functions.zeros_like)
def _(a: Union[np.ndarray, np.number]) -> np.ndarray:
    return np.zeros_like(a)


@registry_numpy_types(functions.stack)
def _(x: Union[np.ndarray, np.number], axis: int = 0) -> List[np.ndarray]:
    return np.stack(x, axis=axis)


@registry_numpy_types(functions.unstack)
def _(x: Union[np.ndarray, np.number], axis: int = 0) -> List[np.ndarray]:
    return [np.squeeze(e, axis) for e in np.split(x, x.shape[axis], axis=axis)]


@registry_numpy_types(functions.moveaxis)
def _(a: np.ndarray, source: Union[int, List[int]], destination: Union[int, List[int]]) -> np.ndarray:
    return np.moveaxis(a, source, destination)


@registry_numpy_types(functions.mean)
def _(a: Union[np.ndarray, np.number], axis: Union[int, List[int]] = None, keepdims: bool = False) -> np.ndarray:
    return np.mean(a, axis=axis, keepdims=keepdims)


@registry_numpy_types(functions.round)
def _(a: Union[np.ndarray, np.number], decimals: int = 0) -> np.ndarray:
    return np.round(a, decimals=decimals)
