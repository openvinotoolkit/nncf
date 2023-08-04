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
import abc
from abc import abstractmethod
from typing import Generic
from typing import List, Optional, TypeVar
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")
TensorElementsType = TypeVar("TensorElementsType")


class NNCFTensor(Generic[TensorType], abc.ABC):
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    @property
    @abstractmethod
    def backend(self) -> Type['NNCFTensorBackend']:
        pass

    def __init__(self, tensor: TensorType):
        self._tensor: TensorType = tensor

    def __eq__(self, other: "NNCFTensor") -> bool:
        # Assuming every backend implements this basic semantic
        return self._tensor == other.tensor

    @property
    def tensor(self):
        return self._tensor

    @property
    @abstractmethod
    def shape(self) -> List[int]:
        return self._tensor.shape

    @property
    @abstractmethod
    def device(self) -> DeviceType:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def mean(self, axis: int) -> 'NNCFTensor':
        pass

    @abstractmethod
    def reshape(self, *shape: int) -> 'NNCFTensor':
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        pass


class NNCFTensorBackend(abc.ABC):
    @staticmethod
    @abstractmethod
    def moveaxis(x: NNCFTensor, src: int, dst: int) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def mean(x: NNCFTensor, axis: Union[int, Tuple[int, ...]]) -> NNCFTensor:
        pass

    @staticmethod
    @abstractmethod
    def mean_of_list(tensor_list: List[NNCFTensor], axis: int) -> NNCFTensor:
        pass
