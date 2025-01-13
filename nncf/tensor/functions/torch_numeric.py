# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from nncf.tensor import TensorDataType
from nncf.tensor import TensorDeviceType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TypeInfo
from nncf.tensor.functions import numeric as numeric
from nncf.tensor.tensor import TTensor

DTYPE_MAP = {
    TensorDataType.float16: torch.float16,
    TensorDataType.bfloat16: torch.bfloat16,
    TensorDataType.float32: torch.float32,
    TensorDataType.float64: torch.float64,
    TensorDataType.int8: torch.int8,
    TensorDataType.int32: torch.int32,
    TensorDataType.int64: torch.int64,
    TensorDataType.uint8: torch.uint8,
}

DEVICE_MAP = {TensorDeviceType.CPU: "cpu", TensorDeviceType.GPU: "cuda"}

DTYPE_MAP_REV = {v: k for k, v in DTYPE_MAP.items()}
DEVICE_MAP_REV = {v: k for k, v in DEVICE_MAP.items()}


def convert_to_torch_device(device: TensorDeviceType) -> str:
    return DEVICE_MAP[device] if device is not None else None


def convert_to_torch_dtype(dtype: TensorDataType) -> torch.dtype:
    return DTYPE_MAP[dtype] if dtype is not None else None


@numeric.device.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorDeviceType:
    return DEVICE_MAP_REV[a.device.type]


@numeric.backend.register(torch.Tensor)
def _(a: torch.Tensor) -> TensorBackend:
    return TensorBackend.torch


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
    return torch.allclose(a, b.to(dtype=a.dtype), rtol=rtol, atol=atol, equal_nan=equal_nan)


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
        b = torch.tensor(b, device=a.device, dtype=a.dtype)
    return torch.isclose(a, b.to(dtype=a.dtype), atol=atol, rtol=rtol, equal_nan=equal_nan)


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


@numeric.concatenate.register(torch.Tensor)
def _(x: List[torch.Tensor], axis: int = 0) -> List[torch.Tensor]:
    return torch.concatenate(x, dim=axis)


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
    dtype = convert_to_torch_dtype(dtype)
    return torch.mean(a, dim=axis, keepdim=keepdims, dtype=dtype)


@numeric.median.register(torch.Tensor)
def _(
    a: torch.Tensor,
    axis: Union[int, Tuple[int, ...]] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    # See https://github.com/pytorch/pytorch/issues/61582
    if not isinstance(axis, int):
        device = a.device
        result = torch.tensor(np.median(a.detach().cpu().numpy(), axis=axis, keepdims=keepdims))
        return result.type(a.dtype).to(device)
    return quantile(a, q=0.5, axis=axis, keepdims=keepdims)


@numeric.round.register(torch.Tensor)
def _(a: torch.Tensor, decimals=0) -> torch.Tensor:
    return torch.round(a, decimals=decimals)


@numeric.power.register(torch.Tensor)
def _(a: torch.Tensor, exponent: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.pow(a, exponent=exponent)


@numeric.quantile.register(torch.Tensor)
def quantile(
    a: torch.Tensor,
    q: Union[float, List[float]],
    axis: Optional[Union[int, Tuple[int]]] = None,
    keepdims: bool = False,
) -> torch.Tensor:
    device = a.device
    # See https://github.com/pytorch/pytorch/issues/61582
    # https://github.com/pytorch/pytorch/issues/64947
    if a.numel() <= 16_000_000 and isinstance(axis, int) and a.dtype in [torch.float32, torch.float64]:
        return torch.quantile(
            a.to(dtype=torch.float64),
            torch.tensor(q, dtype=torch.float64, device=a.device),
            axis,
            keepdims,
        )
    return torch.tensor(
        np.quantile(a.detach().cpu().numpy().astype(np.float64, copy=False), q=q, axis=axis, keepdims=keepdims)
    ).to(device)


@numeric.percentile.register(torch.Tensor)
def _(
    a: torch.Tensor,
    q: Union[float, List[float]],
    axis: Union[int, Tuple[int, ...], List[int]],
    keepdims: bool = False,
) -> List[Union[torch.Tensor, np.generic]]:
    q = [x / 100 for x in q] if isinstance(q, (list, tuple)) else q / 100
    return numeric.quantile(a, q=q, axis=axis, keepdims=keepdims)


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
    if axes is None:
        return a.t()
    return torch.permute(a, axes)


@numeric.argsort.register(torch.Tensor)
def _(a: torch.Tensor, axis: int = -1, descending=False, stable=False) -> torch.Tensor:
    return torch.argsort(a, dim=axis, descending=descending, stable=stable)


@numeric.diag.register(torch.Tensor)
def _(a: torch.Tensor, k: int = 0) -> torch.Tensor:
    return torch.diag(a, diagonal=k)


@numeric.logical_or.register(torch.Tensor)
def _(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(x1, x2)


@numeric.masked_mean.register(torch.Tensor)
def _(
    x: torch.Tensor, mask: Optional[torch.Tensor], axis: Union[int, Tuple[int, ...], List[int]], keepdims=False
) -> torch.Tensor:
    if mask is None:
        return torch.mean(x, axis=axis, keepdims=keepdims)
    masked_x = x.masked_fill(mask, torch.nan)
    ret = torch.nanmean(masked_x, dim=axis, keepdim=keepdims)
    return torch.nan_to_num(ret)


@numeric.masked_median.register(torch.Tensor)
def _(
    x: torch.Tensor, mask: Optional[torch.Tensor], axis: Union[int, Tuple[int, ...], List[int]], keepdims=False
) -> torch.Tensor:
    if mask is None:
        return numeric.median(x, axis=axis, keepdims=keepdims)

    # See https://github.com/pytorch/pytorch/issues/61582
    if not isinstance(axis, int):
        device = x.device
        masked_x = np.ma.array(x.detach().cpu().numpy(), mask=mask.detach().cpu().numpy())
        result = torch.tensor(np.ma.median(masked_x, axis=axis, keepdims=keepdims))
        return result.type(x.dtype).to(device)
    masked_x = x.masked_fill(mask, torch.nan)
    ret = torch.nanquantile(masked_x, q=0.5, dim=axis, keepdims=keepdims)
    return torch.nan_to_num(ret)


@numeric.expand_dims.register(torch.Tensor)
def _(a: torch.Tensor, axis: Union[int, Tuple[int, ...], List[int]]) -> np.ndarray:
    if type(axis) not in (tuple, list):
        axis = (axis,)

    if len(set(axis)) != len(axis):
        raise ValueError("repeated axis")

    out_ndim = len(axis) + a.dim()

    norm_axis = []
    for ax in axis:
        if ax < -out_ndim or ax >= out_ndim:
            raise ValueError(f"axis {ax} is out of bounds for array of dimension {out_ndim}")
        norm_axis.append(ax + out_ndim if ax < 0 else ax)

    shape_it = iter(a.shape)
    shape = [1 if ax in norm_axis else next(shape_it) for ax in range(out_ndim)]
    return a.reshape(shape)


@numeric.clone.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return a.clone()


@numeric.searchsorted.register(torch.Tensor)
def _(a: torch.Tensor, v: torch.Tensor, side: str = "left", sorter: Optional[torch.Tensor] = None) -> torch.Tensor:
    if side not in ["right", "left"]:
        raise ValueError(f"Invalid value for 'side': {side}. Expected 'right' or 'left'.")
    if a.dim() != 1:
        raise ValueError(f"Input tensor 'a' must be 1-D. Received {a.dim()}-D tensor.")
    return torch.searchsorted(sorted_sequence=a, input=v, right=(side == "right"), sorter=sorter)


def zeros(
    shape: Tuple[int, ...],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    device = convert_to_torch_device(device)
    dtype = convert_to_torch_dtype(dtype)
    return torch.zeros(*shape, dtype=dtype, device=device)


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    device = convert_to_torch_device(device)
    dtype = convert_to_torch_dtype(dtype)
    p_args = (n,) if m is None else (n, m)
    return torch.eye(*p_args, dtype=dtype, device=device)


def arange(
    start: float,
    end: float,
    step: float,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    device = convert_to_torch_device(device)
    dtype = convert_to_torch_dtype(dtype)
    return torch.arange(start, end, step, dtype=dtype, device=device)


def from_numpy(ndarray: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ndarray)


@numeric.log2.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.log2(a)


@numeric.ceil.register(torch.Tensor)
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.ceil(a)


def tensor(
    data: Union[TTensor, Sequence[float]],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    device = convert_to_torch_device(device)
    dtype = convert_to_torch_dtype(dtype)
    return torch.tensor(data, dtype=dtype, device=device)
