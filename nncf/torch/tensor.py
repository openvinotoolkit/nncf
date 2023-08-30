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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import torch

from nncf.common.tensor import DeviceType
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend
from nncf.common.tensor import TensorDtype

_DTYPE_MAP: Dict[TensorDtype, torch.dtype] = {TensorDtype.FLOAT32: torch.float32, TensorDtype.INT64: torch.int64}

_INV_DTYPE_MAP: Dict[torch.dtype, TensorDtype] = {v: k for k, v in _DTYPE_MAP.items()}


class PTNNCFTensor(NNCFTensor[torch.Tensor]):
    @property
    def backend(self) -> Type["NNCFTensorBackend"]:
        return PTNNCFTensorBackend

    def _is_native_bool(self, bool_result: Any) -> bool:
        return False  # TODO (vshampor): check whether this is relevant for PT

    @property
    def ndim(self) -> int:
        return len(self._tensor.shape)

    @property
    def shape(self) -> List[int]:
        return list(self._tensor.shape)

    @property
    def device(self) -> DeviceType:
        return self._tensor.device

    @property
    def size(self) -> int:
        return self._tensor.numel()

    def is_empty(self) -> bool:
        return self._tensor.numel() == 0

    def mean(self, axis: int, keepdims: bool = None) -> "PTNNCFTensor":
        return PTNNCFTensor(self._tensor.mean(dim=axis, keepdim=keepdims))

    def median(self, axis: int = None, keepdims: bool = False) -> "PTNNCFTensor":
        return PTNNCFTensor(self._tensor.median(dim=axis, keepdim=keepdims))

    def reshape(self, *shape: int) -> "PTNNCFTensor":
        return PTNNCFTensor(self._tensor.reshape(*shape))

    def to_numpy(self) -> np.ndarray:
        return self._tensor.numpy(force=True)  # force=True to automatically get a CPU copy if the tensor is not on CPU

    def matmul(self, other: "PTNNCFTensor") -> "PTNNCFTensor":
        return PTNNCFTensor(self._tensor.matmul(other._tensor))

    def astype(self, dtype: TensorDtype) -> "PTNNCFTensor":
        return PTNNCFTensor(self._tensor.to(dtype=_DTYPE_MAP[dtype]))

    @property
    def dtype(self) -> TensorDtype:
        return _INV_DTYPE_MAP[self._tensor.dtype]

    def any(self) -> bool:
        return bool(self._tensor.any())

    def all(self) -> bool:
        return bool(self._tensor.all())

    def min(self) -> float:
        return self._tensor.min().item()

    def max(self) -> float:
        return self._tensor.max().item()

    def flatten(self) -> "PTNNCFTensor":
        return PTNNCFTensor(self._tensor.flatten())


class PTNNCFTensorBackend(NNCFTensorBackend):
    inf = torch.inf
    
    @staticmethod
    def moveaxis(tensor: PTNNCFTensor, src: int, dst: int) -> PTNNCFTensor:
        return PTNNCFTensor(torch.moveaxis(tensor.tensor, source=src, destination=dst))

    @staticmethod
    def mean(tensor: PTNNCFTensor, axis: Union[int, Tuple[int, ...]], keepdims: bool = False) -> PTNNCFTensor:
        return PTNNCFTensor(torch.mean(tensor.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def mean_of_list(tensor_list: List[PTNNCFTensor], axis: int) -> PTNNCFTensor:
        cated = torch.concat([t.tensor for t in tensor_list])
        return PTNNCFTensor(torch.mean(cated, dim=axis))

    @staticmethod
    def isclose_all(tensor1: PTNNCFTensor, tensor2: PTNNCFTensor, rtol=1e-05, atol=1e-08) -> bool:
        return bool(torch.isclose(tensor1.tensor, tensor2.tensor, rtol=rtol, atol=atol).all())

    @staticmethod
    def stack(tensor_list: List[PTNNCFTensor]) -> PTNNCFTensor:
        return PTNNCFTensor(torch.stack([t.tensor for t in tensor_list]))

    @staticmethod
    def count_nonzero(tensor: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.count_nonzero(tensor.tensor))

    @staticmethod
    def abs(tensor: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.abs(tensor.tensor))

    @staticmethod
    def min(tensor: PTNNCFTensor, axis: int = None) -> PTNNCFTensor:
        return PTNNCFTensor(torch.min(tensor.tensor, dim=axis))

    @staticmethod
    def max(tensor: PTNNCFTensor, axis: int = None) -> PTNNCFTensor:
        return PTNNCFTensor(torch.max(tensor.tensor, dim=axis))

    @staticmethod
    def min_of_list(tensor_list: List[PTNNCFTensor], axis: int = None) -> PTNNCFTensor:
        cated = torch.concat([t.tensor for t in tensor_list])
        return PTNNCFTensor(torch.min(cated, dim=axis))

    @staticmethod
    def max_of_list(tensor_list: List[PTNNCFTensor], axis: int = None) -> PTNNCFTensor:
        cated = torch.concat([t.tensor for t in tensor_list])
        return PTNNCFTensor(torch.max(cated, dim=axis))

    @staticmethod
    def expand_dims(tensor: PTNNCFTensor, axes: List[int]) -> PTNNCFTensor:
        assert axes
        sorted_axes = sorted(axes)
        tmp = tensor.tensor
        for ax in reversed(sorted_axes):
            tmp = torch.unsqueeze(tmp, dim=ax)
        return PTNNCFTensor(tmp)

    @staticmethod
    def sum(tensor: PTNNCFTensor, axes: List[int]) -> PTNNCFTensor:
        return PTNNCFTensor(torch.sum(tensor.tensor, dim=axes))

    @staticmethod
    def transpose(tensor: PTNNCFTensor, axes: List[int]) -> PTNNCFTensor:
        return PTNNCFTensor(torch.permute(tensor.tensor, dims=axes))

    @staticmethod
    def eps(dtype: TensorDtype) -> float:
        return torch.finfo(_DTYPE_MAP[dtype]).eps

    @staticmethod
    def median(tensor: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.median(tensor.tensor))

    @staticmethod
    def clip(tensor: PTNNCFTensor, min_val: float, max_val: Optional[float] = None) -> PTNNCFTensor:
        return PTNNCFTensor(torch.clip(tensor.tensor, min=min_val, max=max_val))

    @staticmethod
    def ones(shape: Union[int, List[int]], dtype: TensorDtype) -> PTNNCFTensor:
        return PTNNCFTensor(torch.ones(size=shape, dtype=_DTYPE_MAP[dtype]))

    @staticmethod
    def squeeze(tensor: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.squeeze(tensor.tensor))

    @staticmethod
    def power(tensor: PTNNCFTensor, pwr: float) -> PTNNCFTensor:
        return PTNNCFTensor(torch.pow(tensor.tensor, exponent=pwr))

    @staticmethod
    def quantile(tensor: PTNNCFTensor, quantile: Union[float, List[float]], axis: Union[int, List[int]] = None,
                 keepdims: bool = False) -> Union[float, PTNNCFTensor]:
        return PTNNCFTensor(torch.quantile(tensor.tensor, q=quantile, dim=axis, keepdim=keepdims))

    @staticmethod
    def logical_or(tensor1: PTNNCFTensor, tensor2: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.logical_or(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def masked_mean(tensor: PTNNCFTensor, mask: PTNNCFTensor, axis: int = None, keepdims: bool = False) -> PTNNCFTensor:
        masked_tensor = torch.masked.masked_tensor(tensor, mask)
        return PTNNCFTensor(torch.masked.mean(masked_tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def masked_median(tensor: PTNNCFTensor, mask: PTNNCFTensor, axis: int = None, keepdims: bool = False) -> PTNNCFTensor:
        masked_tensor = torch.masked.masked_tensor(tensor, mask)
        return PTNNCFTensor(torch.masked.median(masked_tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def concatenate(tensor_list: List[PTNNCFTensor], axis: int = None) -> PTNNCFTensor:
        return PTNNCFTensor(torch.concatenate([t.tensor for t in tensor_list], dim=axis))

    @staticmethod
    def amin(tensor: PTNNCFTensor, axis: List[int], keepdims: bool = None) -> PTNNCFTensor:
        if keepdims is None:
            keepdims = False
        return PTNNCFTensor(torch.amin(tensor.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def amax(tensor: PTNNCFTensor, axis: List[int], keepdims: bool = None) -> PTNNCFTensor:
        if keepdims is None:
            keepdims = False
        return PTNNCFTensor(torch.amax(tensor.tensor, dim=axis, keepdim=keepdims))

    @staticmethod
    def minimum(tensor1: PTNNCFTensor, tensor2: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.minimum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def maximum(tensor1: PTNNCFTensor, tensor2: PTNNCFTensor) -> PTNNCFTensor:
        return PTNNCFTensor(torch.maximum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def unstack(tensor: PTNNCFTensor, axis: int = 0) -> List[PTNNCFTensor]:
        return [PTNNCFTensor(t) for t in torch.unbind(tensor.tensor, dim=axis)]
