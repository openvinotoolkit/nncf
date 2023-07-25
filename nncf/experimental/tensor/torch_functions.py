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

import torch

from nncf.experimental.tensor import TensorDataType
from nncf.experimental.tensor import TensorDeviceType
from nncf.experimental.tensor import functions

DTYPE_MAP = {
    TensorDataType.float16: torch.float16,
    TensorDataType.float32: torch.float32,
    TensorDataType.float64: torch.float64,
    TensorDataType.int8: torch.int8,
    TensorDataType.uint8: torch.uint8,
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


@functions.device.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDeviceType:
    DEVICE_MAP = {
        "cpu": TensorDeviceType.CPU,
        "cuda": TensorDeviceType.GPU,
    }
    return DEVICE_MAP[a.device.type]


@functions.squeeze.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> torch.Tensor:
    if axis is None:
        return a.squeeze()
    return a.squeeze(axis)


@functions.flatten.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return a.flatten()


@functions.max.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> torch.Tensor:
    if axis is None:
        return torch.max(a)
    return torch.max(a, dim=axis).values


@functions.min.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> torch.Tensor:
    if axis is None:
        return torch.min(a)
    return torch.min(a, dim=axis).values


@functions.abs.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.absolute(a)


@functions.astype.register(torch.Tensor)
def _(a: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
    return a.type(DTYPE_MAP[dtype])


@functions.dtype.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.dtype]


@functions.reshape.register(torch.Tensor)
def _(a: torch.Tensor, shape: List[int]) -> torch.Tensor:
    return a.reshape(shape)


@functions.all.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Union[torch.Tensor, bool]:
    if axis is None:
        return torch.all(a)
    return torch.all(a, dim=axis)


@functions.allclose.register(torch.Tensor)
def _(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> bool:
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@functions.any.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> Union[torch.Tensor, bool]:
    if axis is None:
        return torch.any(a)
    return torch.any(a, dim=axis)


@functions.count_nonzero.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int]]] = None) -> torch.Tensor:
    return torch.count_nonzero(a, dim=axis)


@functions.isempty.register(torch.Tensor)
def _(a: torch.Tensor) -> bool:
    return a.numel() == 0


@functions.isclose.register(torch.Tensor)
def _(a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False):
    return torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


@functions.maximum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.maximum(x1, x2)


@functions.minimum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.minimum(x1, x2)


@functions.ones_like.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(a)


@functions.where.register(torch.Tensor)
def _(
    condition: torch.Tensor, x: Union[torch.Tensor, float, bool], y: Union[torch.Tensor, float, bool]
) -> torch.Tensor:
    return torch.where(condition, x, y)


@functions.zeros_like.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(a)
