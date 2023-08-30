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
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type, Union

import numpy as np
from numpy import dtype
from numpy.ma import MaskedArray
from numpy.ma.core import MaskedConstant

from nncf import TargetDevice
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend
from nncf.common.tensor import TensorDtype

_DTYPE_MAP: Dict[TensorDtype, Any] = {TensorDtype.FLOAT32: np.float32, TensorDtype.INT64: np.int64}

_INV_DTYPE_MAP: Dict[Any, TensorDtype] = {v: k for k, v in _DTYPE_MAP.items()}
_INV_DTYPE_MAP[dtype("float32")] = TensorDtype.FLOAT32


class NPNNCFTensor(NNCFTensor[np.ndarray]):
    """
    Implementation of NNCFTensor over NumPy.
    """

    def _is_native_bool(self, bool_result: Any) -> bool:
        return isinstance(bool_result, np.bool_)

    @property
    def ndim(self) -> int:
        return self._tensor.ndim

    @property
    def size(self) -> int:
        return self._tensor.size


    def matmul(self, other: "NPNNCFTensor") -> "NPNNCFTensor":
        return self.__class__(np.matmul(self._tensor, other._tensor))

    def astype(self, dtype: TensorDtype) -> "NPNNCFTensor":
        return self.__class__(self._tensor.astype(_DTYPE_MAP[dtype]))

    @property
    def dtype(self) -> TensorDtype:
        local_dtype = self._tensor.dtype
        return _INV_DTYPE_MAP[local_dtype]

    def any(self) -> bool:
        return self._tensor.any()

    def all(self) -> bool:
        return self._tensor.all()

    def to_numpy(self) -> np.ndarray:
        return self._tensor

    @property
    def shape(self) -> List[int]:
        return self.tensor.shape

    @property
    def backend(self) -> Type[NNCFTensorBackend]:
        return NPNNCFTensorBackend

    def mean(self, axis: int, keepdims: bool = None) -> "NPNNCFTensor":
        if keepdims is None:
            keepdims = np._NoValue
        return self.__class__(np.mean(self.tensor, axis=axis, keepdims=keepdims))

    def median(self, axis: int = None, keepdims: bool = False) -> "NNCFTensor":
        return self.__class__(np.median(self.tensor, axis=axis, keepdims=keepdims))

    @property
    def device(self):
        return TargetDevice.CPU.value

    def is_empty(self) -> bool:
        return self.tensor.size == 0

    def reshape(self, *shape: Tuple[int, ...]) -> "NPNNCFTensor":
        return self.__class__(self.tensor.reshape(*shape))

    def min(self) -> float:
        return self._tensor.min()

    def max(self) -> float:
        return self._tensor.max()

    def flatten(self) -> "NPNNCFTensor":
        return NPNNCFTensor(self._tensor.flatten())


class NPNNCFTensorBackend(NNCFTensorBackend):
    inf = np.inf

    @staticmethod
    def isclose_all(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor, rtol=1e-05, atol=1e-08) -> bool:
        return bool(np.isclose(tensor1.tensor, tensor2.tensor, rtol, atol))

    @staticmethod
    def stack(tensor_list: List[NPNNCFTensor]) -> NPNNCFTensor:
        return NPNNCFTensor(np.stack([nt.tensor for nt in tensor_list]))

    @staticmethod
    def count_nonzero(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.count_nonzero(tensor.tensor))

    @staticmethod
    def abs(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.abs(tensor.tensor))

    @staticmethod
    def min(tensor: NPNNCFTensor, axis: int = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.min(tensor.tensor, axis=axis))

    @staticmethod
    def max(tensor: NPNNCFTensor, axis: int = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.max(tensor.tensor, axis=axis))

    @staticmethod
    def min_of_list(tensor_list: List[NNCFTensor], axis: int = None) -> NNCFTensor:
        return NPNNCFTensor(np.min([t.tensor for t in tensor_list], axis=axis))

    @staticmethod
    def max_of_list(tensor_list: List[NNCFTensor], axis: int = None) -> NNCFTensor:
        return NPNNCFTensor(np.max([t.tensor for t in tensor_list], axis=axis))

    @staticmethod
    def expand_dims(tensor: NPNNCFTensor, axes: List[int]) -> NPNNCFTensor:
        return NPNNCFTensor(np.expand_dims(tensor.tensor, axis=axes))

    @staticmethod
    def sum(tensor: NPNNCFTensor, axes: List[int]) -> NPNNCFTensor:
        return NPNNCFTensor(np.sum(tensor.tensor, axis=tuple(axes)))

    @staticmethod
    def transpose(tensor: NPNNCFTensor, axes: List[int]) -> NPNNCFTensor:
        return NPNNCFTensor(np.transpose(tensor.tensor, axes=axes))

    @staticmethod
    def eps(dtype: TensorDtype) -> float:
        return np.finfo(_DTYPE_MAP[dtype]).eps

    @staticmethod
    def median(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.median(tensor.tensor))

    @staticmethod
    def clip(tensor: NPNNCFTensor, min_val: float, max_val: Optional[float] = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.clip(tensor.tensor, a_min=min_val, a_max=max_val))

    @staticmethod
    def ones(shape: Union[int, List[int]], dtype: TensorDtype) -> NPNNCFTensor:
        return NPNNCFTensor(np.ones(shape, dtype=_DTYPE_MAP[dtype]))

    @staticmethod
    def squeeze(tensor: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.squeeze(tensor.tensor))

    @staticmethod
    def power(tensor: NPNNCFTensor, pwr: float) -> NPNNCFTensor:
        return NPNNCFTensor(np.power(tensor.tensor, pwr))

    @staticmethod
    def quantile(
        tensor: NPNNCFTensor,
        quantile: Union[float, List[float]],
        axis: Union[int, List[int]] = None,
        keepdims: bool = False,
    ) -> Union[float, NNCFTensor]:
        retval = np.quantile(tensor.tensor, quantile, axis=axis, keepdims=keepdims)
        if not isinstance(quantile, list):
            return retval
        else:
            return NPNNCFTensor(retval)

    @staticmethod
    def mean_of_list(tensor_list: List[NPNNCFTensor], axis: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.mean([x.tensor for x in tensor_list], axis=axis))

    @staticmethod
    def mean(tensor: "NPNNCFTensor", axis: int, keepdims: bool = False) -> NPNNCFTensor:
        return NPNNCFTensor(np.mean(tensor.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def moveaxis(tensor: "NPNNCFTensor", src: int, dst: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.moveaxis(tensor.tensor, src, dst))

    @staticmethod
    def logical_or(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.logical_or(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def masked_mean(tensor: NPNNCFTensor, mask: NPNNCFTensor, axis: int = None, keepdims: bool = False) -> NPNNCFTensor:
        masked_x = np.ma.array(tensor.tensor, mask=mask.tensor)
        result = np.ma.mean(masked_x, axis=axis, keepdims=False)
        if isinstance(result, (MaskedConstant, MaskedArray)):
            result = result.data
        return NPNNCFTensor(result)

    @staticmethod
    def masked_median(
        tensor: NPNNCFTensor, mask: NPNNCFTensor, axis: int = None, keepdims: bool = False
    ) -> NPNNCFTensor:
        masked_x = np.ma.array(tensor.tensor, mask=mask.tensor)
        result = np.ma.median(masked_x, axis=axis, keepdims=False)
        if isinstance(result, (MaskedConstant, MaskedArray)):
            result = result.data
        return NPNNCFTensor(result.data)

    @staticmethod
    def concatenate(tensor_list: List[NPNNCFTensor], axis: int = 0) -> NPNNCFTensor:
        return NPNNCFTensor(np.concatenate([t.tensor for t in tensor_list], axis=axis))

    @staticmethod
    def amin(tensor: NPNNCFTensor, axis: List[int], keepdims: bool = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.amin(tensor.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def amax(tensor: NPNNCFTensor, axis: List[int], keepdims: bool = None) -> NPNNCFTensor:
        return NPNNCFTensor(np.amax(tensor.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def minimum(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.minimum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def maximum(tensor1: NPNNCFTensor, tensor2: NPNNCFTensor) -> NPNNCFTensor:
        return NPNNCFTensor(np.maximum(tensor1.tensor, tensor2.tensor))

    @staticmethod
    def unstack(tensor: NPNNCFTensor, axis: int = 0) -> List[NPNNCFTensor]:
        return [
            NPNNCFTensor(np.squeeze(e, axis)) for e in np.split(tensor.tensor, tensor.tensor.shape[axis], axis=axis)
        ]
