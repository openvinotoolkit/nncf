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
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
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
    a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: Optional[bool] = None
) -> torch.Tensor:
    # Analog of numpy.max is torch.amax
    if keepdim is None:
        keepdim = False
    if axis is None:
        return torch.amax(a, keepdim=keepdim)
    return torch.amax(a, dim=axis, keepdim=keepdim)


@fns.min.register(torch.Tensor)
def _(
    a: torch.Tensor, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdim: Optional[bool] = None
) -> torch.Tensor:
    # Analog of numpy.min is torch.amin
    if keepdim is None:
        keepdim = False
    if axis is None:
        return torch.amin(a, keepdim=keepdim)
    return torch.amin(a, dim=axis, keepdim=keepdim)


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
        x2 = torch.tensor(x2, device=x1.device)
    return torch.maximum(x1, x2)


@fns.minimum.register(torch.Tensor)
def _(x1: torch.Tensor, x2: Union[torch.Tensor, float]) -> torch.Tensor:
    if not isinstance(x2, torch.Tensor):
        x2 = torch.tensor(x2, device=x1.device)
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
def _(x: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
    return torch.stack(x, dim=axis)


@fns.unstack.register(torch.Tensor)
def _(x: torch.Tensor, axis: int = 0) -> List[torch.Tensor]:
    if not list(x.shape):
        x = x.unsqueeze(0)
    return torch.unbind(x, dim=axis)


@fns.moveaxis.register(torch.Tensor)
def _(a: torch.Tensor, source: Union[int, List[int]], destination: Union[int, List[int]]) -> torch.Tensor:
    return torch.moveaxis(a, source, destination)


@fns.mean.register(torch.Tensor)
def _(a: torch.Tensor, axis: Union[int, List[int]] = None, keepdims: Optional[bool] = None) -> torch.Tensor:
    if keepdims is None:
        keepdims = False
    return torch.mean(a, dim=axis, keepdim=keepdims)


@fns.round.register(torch.Tensor)
def _(a: torch.Tensor, decimals=0) -> torch.Tensor:
    return torch.round(a, decimals=decimals)


@fns.binary_operator.register(torch.Tensor)
def _(a: torch.Tensor, b: torch.Tensor, operator_fn: Callable) -> torch.Tensor:
    return operator_fn(a, b)


@fns.binary_reverse_operator.register(torch.Tensor)
def _(a: torch.Tensor, b: torch.Tensor, operator_fn: Callable) -> torch.Tensor:
    return operator_fn(b, a)


@fns.to_numpy.register(torch.Tensor)
def _(a: torch.Tensor) -> np.ndarray:
    return a.numpy(force=True)


@fns.inf.register(torch.Tensor)
def _(a: torch.Tensor) -> Any:
    return torch.inf


def _expand_scalars_in_tensor_list(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    retval = []
    for t in tensor_list:
        if len(t.size()) == 0:
            retval.append(t.reshape(1))
        else:
            retval.append(t)
    return retval


@fns.concatenate.register(torch.Tensor)
def _(x: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
    expanded_scalar_list = _expand_scalars_in_tensor_list(x)
    return torch.concatenate(expanded_scalar_list, dim=axis)


@fns.min_of_list.register(torch.Tensor)
def _(x: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
    cated = torch.concatenate(_expand_scalars_in_tensor_list(x))
    return torch.min(cated, dim=axis).values


@fns.max_of_list.register(torch.Tensor)
def _(x: List[torch.Tensor], axis: int = 0) -> torch.Tensor:
    cated = torch.concatenate(_expand_scalars_in_tensor_list(x))
    return torch.max(cated, dim=axis).values


@fns.amax.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[List[int]] = None, keepdims: Optional[bool] = None) -> torch.Tensor:
    if keepdims is None:
        keepdims = False
    return torch.amax(a, dim=axis, keepdim=keepdims)


@fns.amin.register(torch.Tensor)
def _(a: torch.Tensor, axis: Optional[List[int]] = None, keepdims: Optional[bool] = None) -> torch.Tensor:
    if keepdims is None:
        keepdims = False
    return torch.amin(a, dim=axis, keepdim=keepdims)


@fns.clip.register(torch.Tensor)
def _(a: torch.Tensor, min_val: float, max_val: Optional[float] = None) -> torch.Tensor:
    return torch.clip(a, min=min_val, max=max_val)


@fns.sum.register(torch.Tensor)
def _(a: torch.Tensor, axes: List[int]) -> torch.Tensor:
    return torch.sum(a, dim=axes)


@fns.transpose.register(torch.Tensor)
def _(a: torch.Tensor, axes: List[int]) -> torch.Tensor:
    return torch.permute(a, dims=axes)


@fns.eps.register(torch.Tensor)
def _(a: torch.Tensor, dtype: TensorDataType) -> float:
    return torch.finfo(DTYPE_MAP[dtype]).eps


@fns.median.register(torch.Tensor)
def _(a: torch.Tensor, axis: Union[int, Tuple[int]] = None, keepdims: Optional[bool] = None) -> torch.Tensor:
    if keepdims is None:
        keepdims = False
    return fns.quantile(a, q=0.5, axis=axis, keepdims=keepdims)


@fns.power.register(torch.Tensor)
def _(a: torch.Tensor, pwr: float) -> torch.Tensor:
    return torch.pow(a, exponent=pwr)


@fns.matmul.register(torch.Tensor)
def _(
    a: torch.Tensor,
    b: torch.Tensor,
) -> Union[np.ndarray, np.number]:
    return torch.matmul(a, b)


@fns.quantile.register(torch.Tensor)
def _(
    a: torch.Tensor,
    q: Union[float, List[float]],
    axis: Union[int, List[int]] = None,
    keepdims: Optional[bool] = None,
) -> Union[float, torch.Tensor]:
    if keepdims is None:
        keepdims = False
    if isinstance(q, (list, tuple)):
        q = torch.Tensor(q).to(a.device)
    if not isinstance(axis, (list, tuple)):
        return torch.quantile(a, q=q, dim=axis, keepdim=keepdims)
    if len(axis) == 1:
        return torch.quantile(a, q=q, dim=axis[0], keepdim=keepdims)
    # As of 2.0.1, torch does not support multidim quantile directly.
    t = a
    orig_ndims = len(t.shape)
    sorted_axes = sorted(axis)
    for ax in reversed(sorted_axes):
        t = t.moveaxis(source=ax, destination=-1)
    t = t.flatten(start_dim=-(len(sorted_axes)), end_dim=-1)
    t = torch.quantile(t, q=q, dim=-1, keepdim=keepdims)
    if len(sorted_axes) == orig_ndims:
        # the flattened tensor is 1D, not 0D in this case
        t = t.reshape([1 for _ in range(orig_ndims)])
    else:
        for ax in sorted_axes:
            t = t.unsqueeze(ax)
    return t


@fns.logical_or.register(torch.Tensor)
def _(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
    return torch.logical_or(tensor1, tensor2)


@fns.masked_mean.register(torch.Tensor)
def _(a: torch.Tensor, mask: torch.Tensor, axis: int = None, keepdims: Optional[bool] = None) -> torch.Tensor:
    return torch.masked.mean(torch.masked.masked_tensor(a, mask), dim=axis, keepdim=keepdims)


@fns.masked_median.register(torch.Tensor)
def _(a: torch.Tensor, mask: torch.Tensor, axis: int = None, keepdims: Optional[bool] = None) -> torch.Tensor:
    return torch.masked.median(torch.masked.masked_tensor(a, mask), dim=axis, keepdim=keepdims)


@fns.size.register(torch.Tensor)
def _(a: torch.Tensor) -> int:
    return a.numel()
