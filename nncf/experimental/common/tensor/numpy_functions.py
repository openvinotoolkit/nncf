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

from typing import Optional, Tuple, Union

import numpy as np

from nncf.experimental.common.tensor import functions
from nncf.experimental.common.tensor.enums import TensorDataType
from nncf.experimental.common.tensor.enums import TensorDeviceType

DTYPE_MAP = {
    TensorDataType.float16: np.dtype(np.float16),
    TensorDataType.float32: np.dtype(np.float32),
    TensorDataType.float64: np.dtype(np.float64),
    TensorDataType.int8: np.dtype(np.int8),
    TensorDataType.uint8: np.dtype(np.uint8),
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


@functions.device.register
def _(a: np.ndarray) -> TensorDeviceType:
    return TensorDeviceType.CPU


@functions.squeeze.register
def _(a: np.ndarray, axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.squeeze(a, axis=axis)


@functions.flatten.register
def _(a: np.ndarray) -> np.ndarray:
    return a.flatten()


@functions.max.register
def _(a: np.ndarray, axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:  # pylint: disable=redefined-builtin
    return np.max(a, axis=axis)


@functions.min.register
def _(a: np.ndarray, axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:  # pylint: disable=redefined-builtin
    return np.min(a, axis=axis)


@functions.abs.register
def _(a: np.ndarray) -> np.ndarray:
    return np.absolute(a)


@functions.astype.register
def _(a: np.ndarray, dtype: TensorDataType) -> np.ndarray:
    return a.astype(DTYPE_MAP[dtype])


@functions.dtype.register
def _(a: np.ndarray) -> TensorDataType:
    return DTYPE_MAP_REV[np.dtype(a.dtype)]


@functions.reshape.register
def _(a: np.ndarray, shape: Union[int, Tuple[int]]) -> np.ndarray:
    return a.reshape(shape)


###############################################################################


@functions.all.register
def _(
    a: np.ndarray, axis: Optional[Union[int, Tuple[int]]] = None
) -> Union[np.ndarray, bool]:  # pylint: disable=redefined-builtin
    return np.all(a, axis=axis)


@functions.allclose.register
def _(a: np.ndarray, b: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@functions.any.register
def _(
    a: np.ndarray, axis: Optional[Union[int, Tuple[int]]] = None
) -> Union[np.ndarray, bool]:  # pylint: disable=redefined-builtin
    return np.any(a, axis=axis)


@functions.count_nonzero.register
def _(a: np.ndarray, axis: Optional[Union[int, Tuple[int]]] = None) -> np.ndarray:
    return np.count_nonzero(a, axis=axis)


@functions.isempty.register
def _(a: np.ndarray) -> bool:
    return a.size == 0


@functions.isclose.register
def _(a: np.ndarray, b: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@functions.maximum.register
def _(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.maximum(x1, x2)


@functions.minimum.register
def _(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    return np.minimum(x1, x2)


@functions.ones_like.register
def _(a: np.ndarray) -> np.ndarray:
    return np.ones_like(a)


@functions.where.register
def _(condition: np.ndarray, x: Union[np.ndarray, float, bool], y: Union[np.ndarray, float, bool]) -> np.ndarray:
    return np.where(condition, x, y)


@functions.zeros_like.register
def _(a: np.ndarray) -> np.ndarray:
    return np.zeros_like(a)
