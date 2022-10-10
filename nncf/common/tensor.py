"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from abc import abstractmethod
from typing import TypeVar, List

import numpy as np

TensorType = TypeVar('TensorType')
DeviceType = TypeVar('DeviceType')


class NNCFTensor(object):
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    @property
    @abstractmethod
    def tensor(self) -> TensorType:
        pass

    @property
    @abstractmethod
    def shape(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def device(self) -> DeviceType:
        pass

    @property
    @abstractmethod
    def __eq__(self, other: "NNCFTensor") -> bool:
        pass

    @abstractmethod
    def __add__(self, other: "NNCFTensor") -> "NNCFTensor":
        pass

    @abstractmethod
    def __sub__(self, other: "NNCFTensor") -> "NNCFTensor":
        pass

    @abstractmethod
    def __truediv__(self, other: "NNCFTensor") -> "NNCFTensor":
        pass

    @abstractmethod
    def to_numpy(self) -> np.array:
        pass
