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


from typing import Any, Iterator, List, Optional, Tuple, TypeVar, Union

TensorType = TypeVar("TensorType")
DeviceType = TypeVar("DeviceType")
TensorElementsType = TypeVar("TensorElementsType")

import nncf.common.tensor_ops as tensor_ops


class Tensor:
    """
    An interface of framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, data: Optional[TensorType]):
        self._data = data

    @staticmethod
    def safe_get_tensor_data(obj: Any):
        return obj.data if isinstance(obj, Tensor) else obj

    @property
    def data(self) -> TensorType:
        return self._data

    @property
    def shape(self) -> List[int]:
        if self.data is None:
            raise RuntimeError("Attempt to get shape of empty NNCFTensor")
        return self.data.shape

    def is_empty(self) -> bool:
        return False

    def __bool__(self):
        return bool(self.data)

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __getitem__(self, index: int) -> "Tensor":
        return Tensor(self.data[index])

    # Math operations

    def __add__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.add(self.data, self.safe_get_tensor_data(other)))

    def __radd__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.radd(self.data, self.safe_get_tensor_data(other)))

    def __sub__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.sub(self.data, self.safe_get_tensor_data(other)))

    def __rsub__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.rsub(self.data, self.safe_get_tensor_data(other)))

    def __mul__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.mul(self.data, self.safe_get_tensor_data(other)))

    def __rmul__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.rmul(self.data, self.safe_get_tensor_data(other)))

    def __pow__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.pow(self.data, self.safe_get_tensor_data(other)))

    def __truediv__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.truediv(self.data, self.safe_get_tensor_data(other)))

    def __floordiv__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.floordiv(self.data, self.safe_get_tensor_data(other)))

    def __neg__(self) -> "Tensor":
        return Tensor(tensor_ops.neg(self.data))

    # Comparison operators

    def __lt__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.lt(self.data, self.safe_get_tensor_data(other)))

    def __le__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.lt(self.data, self.safe_get_tensor_data(other)))

    def __eq__(self, other: "Tensor") -> bool:
        return Tensor(tensor_ops.eq(self.data, self.safe_get_tensor_data(other)))

    def __nq__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.nq(self.data, self.safe_get_tensor_data(other)))

    def __gt__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.gt(self.data, self.safe_get_tensor_data(other)))

    def __ge__(self, other: TensorType) -> "Tensor":
        return Tensor(tensor_ops.ge(self.data, self.safe_get_tensor_data(other)))

    # Tensor functions

    @property
    def device(self) -> Optional[DeviceType]:
        return tensor_ops.device(self.data)

    def size(self, axis: Optional[int] = None) -> "Tensor":
        return Tensor(tensor_ops.size(self.data, axis=axis))

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]] = None) -> "Tensor":
        return Tensor(tensor_ops.squeeze(self.data, axis=axis))

    def zeros_like(self) -> "Tensor":
        return Tensor(tensor_ops.zeros_like(self.data))

    def count_nonzero(self, axis: Optional[TensorType] = None) -> "Tensor":
        return Tensor(tensor_ops.count_nonzero(self.data, axis=axis))

    def max(self, axis: Optional[TensorType] = None) -> "Tensor":
        return Tensor(tensor_ops.maximum(self.data, axis=axis))

    def min(self, axis: Optional[TensorType] = None) -> "Tensor":
        return Tensor(tensor_ops.minimum(self.data, axis=axis))

    def abs(self) -> "Tensor":
        return Tensor(tensor_ops.absolute(self.data))
