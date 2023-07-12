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

import torch

from nncf.common.tensor_new.enums import TensorDataType

DTYPE_MAP = {
    TensorDataType.float16: torch.float16,
    TensorDataType.float32: torch.float32,
    TensorDataType.float64: torch.float64,
    TensorDataType.int8: torch.int8,
    TensorDataType.uint8: torch.uint8,
}


def as_type(target: torch.Tensor, dtype: TensorDataType):
    return target.type(DTYPE_MAP[dtype])


def check_tensor_backend(target: Any):
    return isinstance(target, torch.Tensor)


def device(target: torch.Tensor) -> torch.device:
    return target.device


def squeeze(target: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> torch.Tensor:
    if axis is None:
        return target.squeeze()
    return target.squeeze(axis)


def zeros_like(target: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(target)


def ones_like(target: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(target)


def count_nonzero(target, axis: Optional[int] = None) -> torch.Tensor:
    return torch.count_nonzero(target, dim=axis)


def max(target: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    if axis is None:
        return torch.max(target)
    return torch.tensor(torch.max(target, dim=axis).values)


def min(target: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
    if axis is None:
        return torch.min(target)
    return torch.tensor(torch.min(target, dim=axis).values)


def absolute(target: torch.Tensor) -> torch.Tensor:
    return torch.absolute(target)


def maximum(target: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=target.data.device)
    return torch.maximum(target, other)


def minimum(target: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=target.data.device)
    return torch.minimum(target, other)


def all(target: torch.Tensor) -> bool:
    return torch.all(target)


def any(target: torch.Tensor) -> bool:
    return torch.any(target)


def where(condition: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.where(condition, x, y)


def is_empty(target: torch.Tensor) -> bool:
    return target.numel() == 0


def isclose(a: torch.Tensor, b: torch.Tensor, atol: float = None, equal_nan: bool = False):
    return torch.isclose(a, b, atol=atol, equal_nan=equal_nan)
