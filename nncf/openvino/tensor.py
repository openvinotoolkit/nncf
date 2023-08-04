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
from typing import List, Tuple, Type

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend
from nncf.parameters import TargetDevice


class OVNNCFTensor(NNCFTensor[np.ndarray]):
    """
    A realisation of OpenVINO tensor wrapper for common NNCF algorithms.
    """

    def to_numpy(self) -> np.ndarray:
        return self._tensor

    @property
    def shape(self) -> List[int]:
        return self.tensor.shape

    @property
    def backend(self) -> Type:
        return OVNNCFTensorBackend

    def mean(self, axis: int) -> "OVNNCFTensor":
        return OVNNCFTensor(np.mean(self.tensor, axis))

    @property
    def device(self):
        return TargetDevice.CPU.value

    def is_empty(self) -> bool:
        return self.tensor.size == 0

    def reshape(self, *shape: Tuple[int, ...]) -> "OVNNCFTensor":
        return OVNNCFTensor(self.tensor.reshape(*shape))


class OVNNCFTensorBackend(NNCFTensorBackend):
    @staticmethod
    def mean_of_list(tensor_list: List[NNCFTensor], axis: int) -> OVNNCFTensor:
        return OVNNCFTensor(np.mean([x.tensor for x in tensor_list], axis=axis))

    @staticmethod
    def mean(x: "OVNNCFTensor", axis: int) -> OVNNCFTensor:
        return OVNNCFTensor(np.mean(x.tensor, axis))

    @staticmethod
    def moveaxis(x: "OVNNCFTensor", src: int, dst: int) -> OVNNCFTensor:
        return OVNNCFTensor(np.moveaxis(x.tensor, src, dst))
