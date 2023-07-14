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

from nncf.common.tensor_new.enums import TensorDataType

TensorType = TypeVar("TensorType")


def check_tensor_backend(a: Any) -> bool:
    """
    Return True if module 'numpy_ops.py' can works with type of a.

    :param a: The input to check.
    :return: True if the input is a tensor backend, False otherwise.
    """
    return isinstance(a, (np.ndarray, float, int, list))


############################################
# Tensor methods
############################################

DTYPE_MAP = {
    TensorDataType.float16: np.float16,
    TensorDataType.float32: np.float32,
    TensorDataType.float64: np.float64,
    TensorDataType.int8: np.int8,
    TensorDataType.uint8: np.uint8,
}


def as_type(a: np.ndarray, dtype: TensorDataType):
    return a.astype(DTYPE_MAP[dtype])


def device(a: TensorType) -> None:
    return None


def is_empty(a: np.ndarray) -> bool:
    return a.size == 0


def flatten(a: np.ndarray) -> np.ndarray:
    return a.flatten()


############################################
# Module functions
############################################


def absolute(a: TensorType) -> TensorType:
    return np.absolute(a)


def all(a: TensorType, axis: Optional[TensorType] = None) -> TensorType:  # pylint: disable=redefined-builtin
    return np.all(a, axis=axis)


def allclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def any(a: TensorType, axis: Optional[TensorType] = None) -> TensorType:  # pylint: disable=redefined-builtin
    return np.any(a, axis=axis)


def count_nonzero(a, axis: Optional[TensorType] = None) -> TensorType:
    return np.count_nonzero(a, axis=axis)


def isclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def squeeze(a: TensorType, axis: Optional[Union[int, Tuple[int]]] = None) -> TensorType:
    return np.squeeze(a, axis=axis)


def zeros_like(a: TensorType) -> TensorType:
    return np.zeros_like(a)


def ones_like(a: TensorType) -> TensorType:
    return np.ones_like(a)


def max(a: TensorType, axis: Optional[TensorType] = None) -> TensorType:  # pylint: disable=redefined-builtin
    return np.max(a, axis=axis)


def min(a: TensorType, axis: Optional[TensorType] = None) -> TensorType:  # pylint: disable=redefined-builtin
    return np.min(a, axis=axis)


def maximum(x1: TensorType, x2: TensorType) -> TensorType:
    return np.maximum(x1, x2)


def minimum(x1: TensorType, x2: TensorType) -> TensorType:
    return np.minimum(x1, x2)


def where(condition: np.ndarray, x: Union[np.ndarray, float, bool], y: Union[np.ndarray, float, bool]) -> np.ndarray:
    return np.where(condition, x, y)
