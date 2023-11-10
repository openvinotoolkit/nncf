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
from nncf.experimental.tensor import functions as fns

DTYPE_MAP = {
    TensorDataType.float16: torch.float16,
    TensorDataType.float32: torch.float32,
    TensorDataType.float64: torch.float64,
    TensorDataType.int8: torch.int8,
    TensorDataType.uint8: torch.uint8,
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


@fns.device.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDeviceType:
    DEVICE_MAP = {
        "cpu": TensorDeviceType.CPU,
        "cuda": TensorDeviceType.GPU,
    }
    return DEVICE_MAP[a.device.type]


@fns.squeeze.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    if axis is None:
        return a.squeeze()
    if isinstance(axis, Tuple) and any(1 != a.shape[i] for i in axis):
        # Make Numpy behavior, torch.squeeze skips axes that are not equal to one..
        raise ValueError("Cannot select an axis to squeeze out which has size not equal to one")
    return a.squeeze(axis)


@fns.flatten.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return a.flatten()


@fns.max.register(torch.Tensor)
def _(
    a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False
) -> torch.Tensor:
    return torch.amax(a, dim=axis, keepdim=keepdims)


@fns.min.register(torch.Tensor)
def _(
    a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False
) -> torch.Tensor:
    return torch.amin(a, dim=axis, keepdim=keepdims)


@fns.abs.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.absolute(a)


@fns.astype.register(torch.Tensor)
def _(a: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
    return a.type(DTYPE_MAP[dtype])


@fns.dtype.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.dtype]


@fns.reshape.register(torch.Tensor)
def _(a: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return a.reshape(shape)


@fns.all.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[torch.Tensor, bool]:
    if axis is None:
        return torch.all(a)
    return torch.all(a, dim=axis)


@fns.allclose.register(torch.Tensor)
def _(
    a: torch.Tensor, b: Union[torch.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@fns.any.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Union[torch.Tensor, bool]:
    if axis is None:
        return torch.any(a)
    return torch.any(a, dim=axis)


@fns.count_nonzero.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    return torch.count_nonzero(a, dim=axis)


@fns.isempty.register(torch.Tensor)
def _(a: torch.Tensor) -> bool:
    return a.numel() == 0


@fns.isclose.register(torch.Tensor)
def _(
    a: torch.Tensor, b: Union[torch.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
):
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device)
    return torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


@fns.maximum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.maximum(x1, x2)


@fns.minimum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.minimum(x1, x2)


@fns.ones_like.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(a)


@fns.where.register(torch.Tensor)
def _(
    condition: torch.Tensor, x: Union[torch.Tensor, float, bool], y: Union[torch.Tensor, float, bool]
) -> torch.Tensor:
    return torch.where(condition, x, y)


@fns.zeros_like.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(a)


@fns.stack.register(torch.Tensor)
def _(x: List[torch.Tensor], axis: int = 0) -> List[torch.Tensor]:
    return torch.stack(x, dim=axis)


@fns.unstack.register(torch.Tensor)
def _(x: torch.Tensor, axis: int = 0) -> List[torch.Tensor]:
    if not list(x.shape):
        x = x.unsqueeze(0)
    return torch.unbind(x, dim=axis)


@fns.moveaxis.register(torch.Tensor)
def _(a: torch.Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> torch.Tensor:
    return torch.moveaxis(a, source, destination)


@fns.mean.register(torch.Tensor)
def _(a: torch.Tensor, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> torch.Tensor:
    return torch.mean(a, axis=axis, keepdims=keepdims)


@fns.round.register(torch.Tensor)
def _(a: torch.Tensor, decimals=0) -> torch.Tensor:
    return torch.round(a, decimals=decimals)
