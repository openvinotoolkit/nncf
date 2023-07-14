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

from typing import Any, Optional, Tuple, Union

import torch

from nncf.common.tensor_new.enums import TensorDataType


def check_tensor_backend(a: Any):
    """
    Return True if module 'numpy_ops.py' can works with type of a.

    :param a: The input to check.
    :return: True if the input is a tensor backend, False otherwise.
    """
    return isinstance(a, torch.Tensor)


############################################
# Tensor methods
############################################

DTYPE_MAP = {
    TensorDataType.float16: torch.float16,
    TensorDataType.float32: torch.float32,
    TensorDataType.float64: torch.float64,
    TensorDataType.int8: torch.int8,
    TensorDataType.uint8: torch.uint8,
}


def as_type(a: torch.Tensor, dtype: TensorDataType):
    return a.type(DTYPE_MAP[dtype])


def device(target: torch.Tensor) -> torch.device:
    return target.device


def is_empty(target: torch.Tensor) -> bool:
    return target.numel() == 0


def flatten(a: torch.Tensor) -> torch.Tensor:
    return a.flatten()


############################################
# Module functions
############################################


def absolute(a: torch.Tensor) -> torch.Tensor:
    return torch.absolute(a)


def all(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> bool:  # pylint: disable=redefined-builtin
    if axis is None:
        return torch.all(a)
    return torch.all(a, dim=axis)


def allclose(
    a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


def any(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> bool:  # pylint: disable=redefined-builtin
    if axis is None:
        return torch.any(a)
    return torch.any(a, dim=axis)


def count_nonzero(a, axis: Optional[int] = None) -> torch.Tensor:
    return torch.count_nonzero(a, dim=axis)


def isclose(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    return torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


def max(a: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:  # pylint: disable=redefined-builtin
    if axis is None:
        return torch.max(a)
    return torch.tensor(torch.max(a, dim=axis).values)


def maximum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.maximum(x1, x2)


def min(a: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:  # pylint: disable=redefined-builtin
    if axis is None:
        return torch.min(a)
    return torch.tensor(torch.min(a, dim=axis).values)


def minimum(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.minimum(x1, x2)


def ones_like(a: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(a)


def squeeze(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> torch.Tensor:
    if axis is None:
        return a.squeeze()
    return a.squeeze(axis)


def where(
    condition: torch.Tensor, x: Union[torch.Tensor, float, bool], y: Union[torch.Tensor, float, bool]
) -> torch.Tensor:
    return torch.where(condition, x, y)


def zeros_like(a: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(a)
