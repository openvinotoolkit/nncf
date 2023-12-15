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
from __future__ import annotations

from typing import Any, Optional, Tuple, TypeVar, Union

from nncf.experimental.tensor.enums import TensorDataType
from nncf.experimental.tensor.enums import TensorDeviceType

TTensor = TypeVar("TTensor")


class Tensor:
    """
    An interface to framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, data: TTensor):
        self._data = data.data if isinstance(data, Tensor) else data

    @property
    def data(self) -> TTensor:
        return self._data

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def device(self) -> TensorDeviceType:
        return _call_function("device", self)

    @property
    def dtype(self) -> TensorDeviceType:
        return _call_function("dtype", self)

    def __bool__(self) -> bool:
        return bool(self.data)

    def __iter__(self):
        return TensorIterator(self.data)

    def __getitem__(self, index: int) -> Tensor:
        return Tensor(self.data[index])

    def __str__(self) -> str:
        return f"nncf.Tensor({str(self.data)})"

    def __repr__(self) -> str:
        return f"nncf.Tensor({repr(self.data)})"

    # built-in operations

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data + unwrap_tensor_data(other))

    def __radd__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) + self.data)

    def __sub__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data - unwrap_tensor_data(other))

    def __rsub__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) - self.data)

    def __mul__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data * unwrap_tensor_data(other))

    def __rmul__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) * self.data)

    def __pow__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data ** unwrap_tensor_data(other))

    def __truediv__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data / unwrap_tensor_data(other))

    def __rtruediv__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(other / self.data)

    def __floordiv__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data // unwrap_tensor_data(other))

    def __rfloordiv__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(other // self.data)

    def __neg__(self) -> Tensor:
        return Tensor(-self.data)

    # Comparison operators

    def __lt__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data < unwrap_tensor_data(other))

    def __le__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data <= unwrap_tensor_data(other))

    def __eq__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data == unwrap_tensor_data(other))

    def __ne__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data != unwrap_tensor_data(other))

    def __gt__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data > unwrap_tensor_data(other))

    def __ge__(self, other: Union[Tensor, float]) -> Tensor:
        return Tensor(self.data >= unwrap_tensor_data(other))

    # Tensor functions

    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Tensor:
        return _call_function("squeeze", self, axis)

    def flatten(self) -> Tensor:
        return _call_function("flatten", self)

    def max(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False) -> Tensor:
        return _call_function("max", self, axis, keepdims)

    def min(self, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: Optional[bool] = False) -> Tensor:
        return _call_function("min", self, axis, keepdims)

    def abs(self) -> Tensor:
        return _call_function("abs", self)

    def isempty(self) -> bool:
        return _call_function("isempty", self)

    def astype(self, dtype: TensorDataType) -> Tensor:
        return _call_function("astype", self, dtype)

    def reshape(self, shape: Tuple[int, ...]) -> Tensor:
        return _call_function("reshape", self, shape)


def _call_function(func_name: str, *args):
    """
    Call function from functions.py to avoid circular imports.

    :param func_name: Name of function.
    :return: Result of function call.
    """
    from nncf.experimental.tensor import functions

    fn = getattr(functions, func_name)
    return fn(*args)


class TensorIterator:
    """Iterator for Tensor class"""

    def __init__(self, tensor):
        self._tensor = tensor
        self._index = 0

    def __next__(self) -> Tensor:
        if self._index < len(self._tensor):
            result = self._tensor[self._index]
            self._index += 1
            return Tensor(result)

        raise StopIteration


def unwrap_tensor_data(obj: Any) -> TTensor:
    """
    Return the data of a Tensor object, or the object itself if it is not a Tensor.

    :param obj: The object to unwrap.
    :return: The data of the Tensor object, or the object itself.
    """
    return obj.data if isinstance(obj, Tensor) else obj
