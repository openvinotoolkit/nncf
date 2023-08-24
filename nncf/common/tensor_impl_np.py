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
from typing import Callable
from typing import List
from typing import Tuple
from typing import Type

import numpy as np

from nncf import TargetDevice
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend


class NPNNCFTensor(NNCFTensor[np.ndarray]):
    """
    Implementation of NNCFTensor over NumPy.
    """

    def to_numpy(self) -> np.ndarray:
        return self._tensor

    @property
    def shape(self) -> List[int]:
        return self.tensor.shape

    @property
    def backend(self) -> Type[NNCFTensorBackend]:
        return NPNNCFTensorBackend

    def mean(self, axis: int) -> "NPNNCFTensor":
        return self.__class__(np.mean(self.tensor, axis))

    @property
    def device(self):
        return TargetDevice.CPU.value

    def is_empty(self) -> bool:
        return self.tensor.size == 0

    def reshape(self, *shape: Tuple[int, ...]) -> "NPNNCFTensor":
        return self.__class__(self.tensor.reshape(*shape))


class NPNNCFTensorBackend(NNCFTensorBackend):
    @staticmethod
    def mean_of_list(tensor_list: List[NNCFTensor], axis: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.mean([x.tensor for x in tensor_list], axis=axis))

    @staticmethod
    def mean(x: "NPNNCFTensor", axis: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.mean(x.tensor, axis))

    @staticmethod
    def moveaxis(x: "NPNNCFTensor", src: int, dst: int) -> NPNNCFTensor:
        return NPNNCFTensor(np.moveaxis(x.tensor, src, dst))