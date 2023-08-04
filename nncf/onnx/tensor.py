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
from typing import List
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import NNCFTensorBackend


class ONNXNNCFTensor(NNCFTensor[np.ndarray]):
    """
    A realisation of ONNX tensors wrapper for common NNCF algorithms.
    """

    @property
    def backend(self) -> Type['NNCFTensorBackend']:
        return ONNXNNCFTensorBackend

    @property
    def shape(self) -> List[int]:
        return list(self._tensor.shape)

    def is_empty(self) -> bool:
        return self._tensor.size == 0

    def mean(self, axis: int) -> 'ONNXNNCFTensor':
        return ONNXNNCFTensor(self._tensor.mean(axis))

    def reshape(self, *shape: int) -> 'ONNXNNCFTensor':
        return ONNXNNCFTensor(self._tensor.reshape(*shape))

    def to_numpy(self) -> np.ndarray:
        return self._tensor

    @property
    def device(self):
        return "CPU"


class ONNXNNCFTensorBackend(NNCFTensorBackend):
    @staticmethod
    def moveaxis(x: ONNXNNCFTensor, src: int, dst: int) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(np.moveaxis(x.tensor, src, dst))

    @staticmethod
    def mean(x: ONNXNNCFTensor, axis: Union[int, Tuple[int, ...]]) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis))

    @staticmethod
    def mean_of_list(tensor_list: List[ONNXNNCFTensor], axis: int) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(np.mean([x.tensor for x in tensor_list], axis=axis))

