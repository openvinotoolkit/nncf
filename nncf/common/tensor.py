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

from abc import abstractmethod
from typing import Any, List, Optional, Tuple, TypeVar, Union

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")
TensorElementsType = TypeVar("TensorElementsType")


class NNCFTensor:
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, tensor: Optional[TensorType]):
        self._tensor = tensor

    def __eq__(self, other: "NNCFTensor") -> bool:
        return self._tensor == other.tensor

    @property
    def tensor(self) -> TensorType:
        return self._tensor

    @property
    def shape(self) -> List[int]:
        if self._tensor is None:
            raise RuntimeError("Attempt to get shape of empty NNCFTensor")
        return self._tensor.shape

    @property
    def device(self) -> Optional[DeviceType]:
        return None

    def is_empty(self) -> bool:
        return False


class NNCFTensorExt(NNCFTensor):
    # Build-in math operators

    def __add__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.__class__(self.tensor + other_tensor)

    def __radd__(self, other: Any) -> "NNCFTensor":
        return self.__class__(other + self.tensor)

    def __sub__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.__class__(self.tensor - other_tensor)

    def __rsub__(self, other: Any) -> "NNCFTensor":
        return self.__class__(other - self.tensor)

    def __mul__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.__class__(self.tensor * other_tensor)

    def __rmul__(self, other: Any) -> "NNCFTensor":
        return self.__class__(other * self.tensor)

    def __pow__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.__class__(self.tensor**other_tensor)

    def __truediv__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.__class__(self.tensor / other_tensor)

    def __floordiv__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.__class__(self.tensor // other_tensor)

    def __neg__(self) -> "NNCFTensor":
        return self.__class__(-self.tensor)

    # Comparison operators

    def __lt__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.tensor < other_tensor

    def __le__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.tensor < other_tensor

    def __nq__(self, other: Any) -> "NNCFTensor":
        return self.tensor != other.tensor

    def __gt__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.tensor > other_tensor

    def __ge__(self, other: Any) -> "NNCFTensor":
        other_tensor = other.tensor if isinstance(other, self.__class__) else other
        return self.tensor >= other_tensor

    # Tensor functions

    @abstractmethod
    def size(self, axis: Optional[int] = None) -> Union[int, Tuple[int]]:
        pass

    @abstractmethod
    def squeeze(self, axis: Optional[Union[int, Tuple[int]]] = None) -> "NNCFTensor":
        pass

    @abstractmethod
    def zeros_like(self) -> "NNCFTensor":
        pass

    @abstractmethod
    def count_nonzero(self, axis: Optional[TensorType] = None) -> "NNCFTensor":
        pass

    @abstractmethod
    def max(self, axis: Optional[TensorType] = None) -> "NNCFTensor":
        pass

    @abstractmethod
    def min(self, axis: Optional[TensorType] = None) -> "NNCFTensor":
        pass

    @abstractmethod
    def abs(self) -> "NNCFTensor":
        pass
