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

from typing import Optional, Tuple, TypeVar, Union

import torch

from nncf.common.tensor import NNCFTensorExt

TensorType = TypeVar("TensorType")


class PTNNCFTensor(NNCFTensorExt):
    """
    A realisation of torch tensors wrapper for common NNCF algorithms.
    """

    def __init__(self, tensor: torch.tensor):
        # In case somebody attempts to wrap
        # tensor twice
        if isinstance(tensor, self.__class__):
            tensor = tensor.tensor

        super().__init__(tensor)

    @property
    def device(self) -> torch.device:
        return self._tensor.device

    def size(self, axis: Optional[int] = None) -> "PTNNCFTensor":
        if axis is None:
            return self.__class__(torch.tensor(self.tensor.size()))
        return self.__class__(torch.tensor(self.tensor.size(dim=axis)))

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]] = None) -> "PTNNCFTensor":
        if axis is None:
            return self.__class__(self.tensor.squeeze())
        return self.__class__(self.tensor.squeeze(axis))

    def zeros_like(self) -> "PTNNCFTensor":
        return self.__class__(torch.zeros_like(self.tensor))

    def count_nonzero(self, axis: Optional[TensorType] = None) -> "PTNNCFTensor":
        return self.__class__(torch.count_nonzero(self.tensor, dim=axis))

    def max(self, axis: Optional[TensorType] = None) -> "PTNNCFTensor":
        if axis is None:
            return self.__class__(torch.max(self.tensor))
        return self.__class__(torch.max(self.tensor, dim=axis).values)

    def min(self, axis: Optional[TensorType] = None) -> "PTNNCFTensor":
        if axis is None:
            return self.__class__(torch.min(self.tensor))
        return self.__class__(torch.min(self.tensor, dim=axis).values)

    def abs(self) -> "PTNNCFTensor":
        return self.__class__(torch.abs(self.tensor))
