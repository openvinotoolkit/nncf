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

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch

from nncf.experimental.tensor import TensorDataType
from nncf.experimental.tensor import TensorDeviceType
from nncf.experimental.tensor.definitions import TypeInfo
from nncf.experimental.tensor.functions import numeric as numeric

DTYPE_MAP = {
    TensorDataType.float16: torch.float16,
    TensorDataType.float32: torch.float32,
    TensorDataType.float64: torch.float64,
    TensorDataType.int8: torch.int8,
    TensorDataType.int32: torch.int32,
    TensorDataType.int64: torch.int64,
    TensorDataType.uint8: torch.uint8,
}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}


@numeric.device.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDeviceType:
    DEVICE_MAP = {
        "cpu": TensorDeviceType.CPU,
        "cuda": TensorDeviceType.GPU,
    }
    return DEVICE_MAP[a.device.type]


@numeric.squeeze.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    if axis is None:
        return a.squeeze()
    if isinstance(axis, Tuple) and any(a.shape[i] != 1 for i in axis):
        # Make Numpy behavior, torch.squeeze skips axes that are not equal to one..
        raise ValueError("Cannot select an axis to squeeze out which has size not equal to one")
    return a.squeeze(axis)


@numeric.flatten.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return a.flatten()


@numeric.max.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> torch.Tensor:
    # Analog of numpy.max is torch.amax
    if axis is None:
        return torch.amax(a)
    return torch.amax(a, dim=axis, keepdim=keepdim)


@numeric.min.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: bool = False) -> torch.Tensor:
    # Analog of numpy.min is torch.amin
    if axis is None:
        return torch.amin(a)
    return torch.amin(a, dim=axis, keepdim=keepdim)


@numeric.abs.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.absolute(a)


@numeric.astype.register(torch.Tensor)
def _(a: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
    return a.type(DTYPE_MAP[dtype])


@numeric.dtype.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.dtype]


@numeric.reshape.register(torch.Tensor)
def _(a: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    return a.reshape(shape)


@numeric.all.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    if axis is None:
        return torch.all(a)
    return torch.all(a, dim=axis)


@numeric.allclose.register(torch.Tensor)
def _(
    a: torch.Tensor, b: Union[torch.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device)
    return torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.any.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    if axis is None:
        return torch.any(a)
    return torch.any(a, dim=axis)


@numeric.count_nonzero.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    return torch.count_nonzero(a, dim=axis)


@numeric.isempty.register(torch.Tensor)
def _(a: torch.Tensor) -> bool:
    return a.numel() == 0


@numeric.isclose.register(torch.Tensor)
def _(
    a: torch.Tensor, b: Union[torch.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> torch.Tensor:
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device)
    return torch.isclose(a, b, atol=atol, rtol=rtol, equal_nan=equal_nan)


@numeric.maximum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.maximum(x1, x2)


@numeric.minimum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.minimum(x1, x2)


@numeric.ones_like.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(a)


@numeric.where.register(torch.Tensor)
def _(
    condition: torch.Tensor, x: Union[torch.Tensor, float, bool], y: Union[torch.Tensor, float, bool]
) -> torch.Tensor:
    return torch.where(condition, x, y)


@numeric.zeros_like.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(a)


@numeric.stack.register(torch.Tensor)
def _(x: List[torch.Tensor], axis: int = 0) -> List[torch.Tensor]:
    return torch.stack(x, dim=axis)


@numeric.unstack.register(torch.Tensor)
def _(x: torch.Tensor, axis: int = 0) -> List[torch.Tensor]:
    if not list(x.shape):
        x = x.unsqueeze(0)
    return torch.unbind(x, dim=axis)


@numeric.moveaxis.register(torch.Tensor)
def _(a: torch.Tensor, source: Union[int, Tuple[int, ...]], destination: Union[int, Tuple[int, ...]]) -> torch.Tensor:
    return torch.moveaxis(a, source, destination)


@numeric.mean.register(torch.Tensor)
def _(
    a: torch.Tensor,
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
    dtype: Optional[TensorDataType] = None,
) -> torch.Tensor:
    dtype = DTYPE_MAP[dtype] if dtype else None
    return torch.mean(a, dim=axis, keepdim=keepdims, dtype=dtype)


@numeric.round.register(torch.Tensor)
def _(a: torch.Tensor, decimals=0) -> torch.Tensor:
    return torch.round(a, decimals=decimals)


@numeric.power.register(torch.Tensor)
def _(a: torch.Tensor, exponent: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.pow(a, exponent=exponent)


@numeric.quantile.register(torch.Tensor)
def _(
    a: torch.Tensor,
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: Optional[bool] = None,
) -> torch.Tensor:
    device = a.device
    # See https://github.com/pytorch/pytorch/issues/61582
    # https://github.com/pytorch/pytorch/issues/64947
    if a.numel() <= 16_000_000 and isinstance(axis, int) and a.dtype in [torch.float32, torch.float64]:
        return torch.quantile(
            a,
            torch.tensor(q, dtype=a.dtype, device=a.device),
            axis,
            keepdims,
        ).type(torch.float64)
    return torch.tensor(np.quantile(a.detach().cpu().numpy(), q=q, axis=axis, keepdims=keepdims)).to(device)


@numeric._binary_op_nowarn.register(torch.Tensor)
def _(a: torch.Tensor, b: Union[torch.Tensor, float], operator_fn: Callable) -> torch.Tensor:
    return operator_fn(a, b)


@numeric._binary_reverse_op_nowarn.register(torch.Tensor)
def _(a: torch.Tensor, b: Union[torch.Tensor, float], operator_fn: Callable) -> torch.Tensor:
    return operator_fn(b, a)


@numeric.clip.register(torch.Tensor)
def _(a: torch.Tensor, a_min: Union[torch.Tensor, float], a_max: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.clip(a, a_min, a_max)


@numeric.finfo.register(torch.Tensor)
def _(a: torch.Tensor) -> TypeInfo:
    ti = torch.finfo(a.dtype)
    return TypeInfo(ti.eps, ti.max, ti.min)


@numeric.as_tensor_like.register(torch.Tensor)
def _(a: torch.Tensor, data: Any) -> torch.Tensor:
    return torch.as_tensor(data, device=a.device)


@numeric.item.register(torch.Tensor)
def _(a: torch.Tensor) -> Union[int, float, bool]:
    return a.item()


@numeric.sum.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> torch.Tensor:
    return torch.sum(a, dim=axis, keepdim=keepdims)


@numeric.multiply.register(torch.Tensor)
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.multiply(x1, x2)


@numeric.var.register(torch.Tensor)
def _(
    a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ddof: int = 0
) -> torch.Tensor:
    return torch.var(a, dim=axis, keepdim=keepdims, correction=ddof)


@numeric.size.register(torch.Tensor)
def _(a: torch.Tensor) -> int:
    return torch.numel(a)


@numeric.matmul.register(torch.Tensor)
def _(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x1, x2)


@numeric.unsqueeze.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> torch.Tensor:
    return torch.unsqueeze(a, dim=axis)


@numeric.transpose.register(torch.Tensor)
def _(a: torch.Tensor, axes: Optional[Tuple[int, ...]] = None) -> torch.Tensor:
    return a.t()


@numeric.argsort.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[int] = None, descending=False, stable=False) -> torch.Tensor:
    return torch.argsort(a, dim=axis, descending=descending, stable=stable)
