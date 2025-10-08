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

from typing import Any, Callable, Literal, Optional, Sequence, Union

import numpy as np
import torch
from numpy.typing import NDArray

from nncf.tensor import TensorDataType
from nncf.tensor import TensorDeviceType
from nncf.tensor.definitions import T_AXIS
from nncf.tensor.definitions import T_NUMBER
from nncf.tensor.definitions import T_SHAPE
from nncf.tensor.definitions import T_SHAPE_ARRAY
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


def convert_to_torch_device(device: Optional[TensorDeviceType]) -> Optional[str]:
    return DEVICE_MAP[device] if device is not None else None


def convert_to_torch_dtype(dtype: Optional[TensorDataType]) -> Optional[torch.dtype]:
    return DTYPE_MAP[dtype] if dtype is not None else None


@numeric.device.register
def _(a: torch.Tensor) -> TensorDeviceType:
    return DEVICE_MAP_REV[a.device.type]


@numeric.backend.register
def _(a: torch.Tensor) -> TensorBackend:
    return TensorBackend.torch


@numeric.bincount.register
def _(a: torch.Tensor, *, weights: Optional[torch.Tensor], minlength: int = 0) -> torch.Tensor:
    return torch.bincount(input=a, weights=weights, minlength=minlength)


@numeric.squeeze.register
def _(a: torch.Tensor, axis: T_AXIS = None) -> torch.Tensor:
    if axis is None:
        return a.squeeze()
    if isinstance(axis, tuple) and any(a.shape[i] != 1 for i in axis):
        # Make Numpy behavior, torch.squeeze skips axes that are not equal to one..
        msg = "Cannot select an axis to squeeze out which has size not equal to one"
        raise ValueError(msg)
    return a.squeeze(axis)


@numeric.flatten.register
def _(a: torch.Tensor) -> torch.Tensor:
    return a.flatten()


@numeric.max.register
def _(a: torch.Tensor, axis: T_AXIS = None, keepdims: bool = False) -> torch.Tensor:
    # Analog of numpy.max is torch.amax
    if axis is None:
        return torch.amax(a)
    return torch.amax(a, dim=axis, keepdim=keepdims)


@numeric.min.register
def _(a: torch.Tensor, axis: T_AXIS = None, keepdims: bool = False) -> torch.Tensor:
    # Analog of numpy.min is torch.amin
    if axis is None:
        return torch.amin(a)
    return torch.amin(a, dim=axis, keepdim=keepdims)


@numeric.abs.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.absolute(a)


@numeric.astype.register
def _(a: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
    return a.type(DTYPE_MAP[dtype])


@numeric.dtype.register
def _(a: torch.Tensor) -> TensorDataType:
    return DTYPE_MAP_REV[a.dtype]


@numeric.repeat.register
def _(a: torch.Tensor, repeats: Union[int, torch.Tensor], *, axis: Optional[int] = None) -> torch.Tensor:
    return torch.repeat_interleave(a, repeats=repeats, dim=axis)


@numeric.reshape.register
def _(a: torch.Tensor, shape: T_SHAPE) -> torch.Tensor:
    return a.reshape(shape)


@numeric.atleast_1d.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.atleast_1d(a)  # type: ignore


@numeric.all.register
def _(a: torch.Tensor, axis: T_AXIS = None) -> torch.Tensor:
    if axis is None:
        return torch.all(a)
    return torch.all(a, dim=axis)


@numeric.allclose.register
def _(
    a: torch.Tensor, b: Union[torch.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device)
    return torch.allclose(a, b.to(dtype=a.dtype), rtol=rtol, atol=atol, equal_nan=equal_nan)


@numeric.any.register
def _(a: torch.Tensor, axis: T_AXIS = None) -> torch.Tensor:
    if axis is None:
        return torch.any(a)
    return torch.any(a, dim=axis)


@numeric.count_nonzero.register
def _(a: torch.Tensor, axis: T_AXIS = None) -> torch.Tensor:
    return torch.count_nonzero(a, dim=axis)


@numeric.histogram.register
def _(
    a: torch.Tensor,
    bins: int,
    *,
    range: Optional[tuple[float, float]] = None,
) -> torch.Tensor:
    if range is None:
        return torch.histc(input=a, bins=bins)
    return torch.histc(input=a, bins=bins, min=range[0], max=range[1])


@numeric.isempty.register
def _(a: torch.Tensor) -> bool:
    return a.numel() == 0


@numeric.isclose.register
def _(
    a: torch.Tensor, b: Union[torch.Tensor, float], rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> torch.Tensor:
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=a.device, dtype=a.dtype)
    return torch.isclose(a, b.to(dtype=a.dtype), atol=atol, rtol=rtol, equal_nan=equal_nan)


@numeric.maximum.register
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.maximum(x1, x2)


@numeric.minimum.register
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.data.device)
    return torch.minimum(x1, x2)


@numeric.ones_like.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(a)


@numeric.where.register
def _(
    condition: torch.Tensor, x: Union[torch.Tensor, float, bool], y: Union[torch.Tensor, float, bool]
) -> torch.Tensor:
    return torch.where(condition, x, y)


@numeric.nonzero.register
def _(condition: torch.Tensor) -> torch.Tensor:
    return torch.nonzero(condition, as_tuple=True)


@numeric.zeros_like.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(a)


@numeric.stack.register
def _(x: Sequence[torch.Tensor], axis: int = 0) -> torch.Tensor:
    if not isinstance(x, (tuple, list)):
        x = list(x)
    return torch.stack(x, dim=axis)


@numeric.concatenate.register
def _(x: list[torch.Tensor], axis: int = 0) -> torch.Tensor:
    return torch.concatenate(x, dim=axis)


@numeric.unstack.register
def _(x: torch.Tensor, axis: int = 0) -> list[torch.Tensor]:
    if not list(x.shape):
        x = x.unsqueeze(0)
    return list(torch.unbind(x, dim=axis))


@numeric.moveaxis.register
def _(a: torch.Tensor, source: T_SHAPE, destination: T_SHAPE) -> torch.Tensor:
    return torch.moveaxis(a, source, destination)  # type: ignore[arg-type]


@numeric.mean.register
def _(
    a: torch.Tensor,
    axis: T_AXIS = None,
    keepdims: bool = False,
    dtype: Optional[TensorDataType] = None,
) -> torch.Tensor:
    pt_dtype = convert_to_torch_dtype(dtype)
    return torch.mean(a, dim=axis, keepdim=keepdims, dtype=pt_dtype)


@numeric.median.register
def median(
    a: torch.Tensor,
    axis: T_AXIS = None,
    keepdims: bool = False,
) -> torch.Tensor:
    # See https://github.com/pytorch/pytorch/issues/61582
    if not isinstance(axis, int):
        device = a.device
        result = torch.tensor(np.median(a.detach().cpu().numpy(), axis=axis, keepdims=keepdims))
        return result.type(a.dtype).to(device)
    return quantile(a, q=0.5, axis=axis, keepdims=keepdims)


@numeric.floor.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.floor(a)


@numeric.round.register
def _(a: torch.Tensor, decimals: int = 0) -> torch.Tensor:
    return torch.round(a, decimals=decimals)


@numeric.power.register
def _(a: torch.Tensor, exponent: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.pow(a, exponent=exponent)


@numeric.quantile.register
def quantile(
    a: torch.Tensor,
    q: Union[float, list[float]],
    axis: T_AXIS = None,
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


@numeric.percentile.register
def _(
    a: torch.Tensor,
    q: Union[float, list[float]],
    axis: T_AXIS,
    keepdims: bool = False,
) -> torch.Tensor:
    q = [x / 100 for x in q] if isinstance(q, (list, tuple)) else q / 100
    return quantile(a, q=q, axis=axis, keepdims=keepdims)


@numeric._binary_op_nowarn.register
def _(a: torch.Tensor, b: Union[torch.Tensor, float], operator_fn: Callable[..., Any]) -> torch.Tensor:
    return operator_fn(a, b)


@numeric._binary_reverse_op_nowarn.register
def _(a: torch.Tensor, b: Union[torch.Tensor, float], operator_fn: Callable[..., Any]) -> torch.Tensor:
    return operator_fn(b, a)


@numeric.clip.register
def _(a: torch.Tensor, a_min: Union[torch.Tensor, float], a_max: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.clip(a, a_min, a_max)  # type: ignore[arg-type]


@numeric.finfo.register
def _(a: torch.Tensor) -> TypeInfo:
    ti = torch.finfo(a.dtype)
    return TypeInfo(ti.eps, ti.max, ti.min)


@numeric.as_tensor_like.register
def _(a: torch.Tensor, data: Any) -> torch.Tensor:
    return torch.as_tensor(data, device=a.device)


@numeric.item.register
def _(a: torch.Tensor) -> T_NUMBER:
    return a.item()


@numeric.sum.register
def _(a: torch.Tensor, axis: T_AXIS = None, keepdims: bool = False) -> torch.Tensor:
    return torch.sum(a, dim=axis, keepdim=keepdims)


@numeric.cumsum.register
def _(a: torch.Tensor, axis: T_AXIS = None) -> torch.Tensor:
    return torch.cumsum(a, dim=axis)


@numeric.multiply.register
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    return torch.multiply(x1, x2)


@numeric.var.register
def _(a: torch.Tensor, axis: T_AXIS = None, keepdims: bool = False, ddof: int = 0) -> torch.Tensor:
    return torch.var(a, dim=axis, keepdim=keepdims, correction=ddof)


@numeric.size.register
def _(a: torch.Tensor) -> int:
    return torch.numel(a)


@numeric.matmul.register
def _(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x1, x2)


@numeric.unsqueeze.register
def _(a: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.unsqueeze(a, dim=axis)


@numeric.transpose.register
def _(a: torch.Tensor, axes: Optional[T_SHAPE_ARRAY] = None) -> torch.Tensor:
    if axes is None:
        return a.t()
    return torch.permute(a, axes)


@numeric.argsort.register
def _(a: torch.Tensor, axis: int = -1, descending: bool = False, stable: bool = False) -> torch.Tensor:
    return torch.argsort(a, dim=axis, descending=descending, stable=stable)


@numeric.diag.register
def _(a: torch.Tensor, k: int = 0) -> torch.Tensor:
    return torch.diag(a, diagonal=k)


@numeric.logical_or.register
def _(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(x1, x2)


@numeric.masked_mean.register
def _(x: torch.Tensor, mask: Optional[torch.Tensor], axis: T_AXIS, keepdims: bool = False) -> torch.Tensor:
    if mask is None:
        return torch.mean(x, dim=axis, keepdim=keepdims)
    masked_x = x.masked_fill(mask, torch.nan)
    ret = torch.nanmean(masked_x, dim=axis, keepdim=keepdims)
    return torch.nan_to_num(ret)


@numeric.masked_median.register
def _(x: torch.Tensor, mask: Optional[torch.Tensor], axis: T_AXIS, keepdims: bool = False) -> torch.Tensor:
    if mask is None:
        return median(x, axis=axis, keepdims=keepdims)

    # See https://github.com/pytorch/pytorch/issues/61582
    if not isinstance(axis, int):
        device = x.device
        np_masked_x = np.ma.array(x.detach().cpu().numpy(), mask=mask.detach().cpu().numpy())  # type: ignore[no-untyped-call]
        result = torch.tensor(np.ma.median(np_masked_x, axis=axis, keepdims=keepdims))  # type: ignore[no-untyped-call]
        return result.type(x.dtype).to(device)
    pt_masked_x = x.masked_fill(mask, torch.nan)
    ret = torch.nanquantile(pt_masked_x, q=0.5, dim=axis, keepdim=keepdims)
    return torch.nan_to_num(ret)


@numeric.expand_dims.register
def _(a: torch.Tensor, axis: T_SHAPE) -> torch.Tensor:
    if not isinstance(axis, (tuple, list)):
        axis = (axis,)

    if len(set(axis)) != len(axis):
        msg = "repeated axis"
        raise ValueError(msg)

    out_ndim = len(axis) + a.dim()

    norm_axis = []
    for ax in axis:
        if ax < -out_ndim or ax >= out_ndim:
            msg = f"axis {ax} is out of bounds for array of dimension {out_ndim}"
            raise ValueError(msg)
        norm_axis.append(ax + out_ndim if ax < 0 else ax)

    shape_it = iter(a.shape)
    shape = [1 if ax in norm_axis else next(shape_it) for ax in range(out_ndim)]
    return a.reshape(shape)


@numeric.clone.register
def _(a: torch.Tensor) -> torch.Tensor:
    return a.clone()


@numeric.searchsorted.register
def _(
    a: torch.Tensor, v: torch.Tensor, side: Literal["left", "right"] = "left", sorter: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if side not in ["right", "left"]:
        msg = f"Invalid value for 'side': {side}. Expected 'right' or 'left'."
        raise ValueError(msg)
    if a.dim() != 1:
        msg = f"Input tensor 'a' must be 1-D. Received {a.dim()}-D tensor."
        raise ValueError(msg)
    return torch.searchsorted(sorted_sequence=a, input=v, right=(side == "right"), sorter=sorter)


def zeros(
    shape: tuple[int, ...],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    pt_device = convert_to_torch_device(device)
    pt_dtype = convert_to_torch_dtype(dtype)
    return torch.zeros(*shape, dtype=pt_dtype, device=pt_device)


def eye(
    n: int,
    m: Optional[int] = None,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    pt_device = convert_to_torch_device(device)
    pt_dtype = convert_to_torch_dtype(dtype)
    p_args = (n,) if m is None else (n, m)
    return torch.eye(*p_args, dtype=pt_dtype, device=pt_device)


def linspace(
    start: float,
    end: float,
    num: int,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    pt_device = convert_to_torch_device(device)
    pt_dtype = convert_to_torch_dtype(dtype)
    return torch.linspace(start, end, num, dtype=pt_dtype, device=pt_device)


def arange(
    start: float,
    end: float,
    step: float,
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    pt_device = convert_to_torch_device(device)
    pt_dtype = convert_to_torch_dtype(dtype)
    return torch.arange(start, end, step, dtype=pt_dtype, device=pt_device)


def from_numpy(ndarray: NDArray[Any]) -> torch.Tensor:
    return torch.from_numpy(ndarray)


@numeric.log2.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.log2(a)


@numeric.ceil.register
def _(a: torch.Tensor) -> torch.Tensor:
    return torch.ceil(a)


def tensor(
    data: Union[TTensor, Sequence[float]],
    *,
    dtype: Optional[TensorDataType] = None,
    device: Optional[TensorDeviceType] = None,
) -> torch.Tensor:
    pt_device = convert_to_torch_device(device)
    pt_dtype = convert_to_torch_dtype(dtype)
    return torch.tensor(data, dtype=pt_dtype, device=pt_device)


@numeric.as_numpy_tensor.register
def _(a: torch.Tensor) -> NDArray[Any]:
    return a.cpu().detach().numpy()
