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

import numpy as np

from nncf.common.tensor import NNCFTensorExt

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")


class NUMPYNNCFTensor(NNCFTensorExt):
    """
    A realisation of numpy tensors wrapper for common NNCF algorithms.
    """

    def __init__(self, tensor: np.ndarray):
        # In case somebody attempts to wrap
        # tensor twice
        if isinstance(tensor, self.__class__):
            tensor = tensor.tensor

        super().__init__(tensor)

    def size(self, axis: Optional[int] = None) -> "NUMPYNNCFTensor":
        if axis is None:
            return self.__class__(np.array(self.tensor.shape))
        return self.__class__(np.array(self.tensor.shape[axis]))

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]] = None) -> "NUMPYNNCFTensor":
        return self.__class__(self.tensor.squeeze(axis))

    def zeros_like(self) -> "NUMPYNNCFTensor":
        return self.__class__(np.zeros_like(self.tensor))

    def count_nonzero(self, axis: Optional[TensorType] = None) -> "NUMPYNNCFTensor":
        return self.__class__(np.count_nonzero(self.tensor, axis=axis))

    def max(self, axis: Optional[TensorType] = None) -> "NUMPYNNCFTensor":
        return self.__class__(np.max(self.tensor, axis=axis))

    def min(self, axis: Optional[TensorType] = None) -> "NUMPYNNCFTensor":
        return self.__class__(np.min(self.tensor, axis=axis))

    def abs(self) -> "NUMPYNNCFTensor":
        return self.__class__(np.abs(self.tensor))
