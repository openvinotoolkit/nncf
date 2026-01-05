# Copyright (c) 2026 Intel Corporation
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

import operator
from typing import Any, Iterator, Optional, Union, cast

import nncf
from nncf.common.utils.backend import BackendType
from nncf.tensor.definitions import T_AXIS
from nncf.tensor.definitions import T_NUMBER
from nncf.tensor.definitions import T_SHAPE
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType
from nncf.tensor.definitions import TensorDeviceType

TTensor = Any


class Tensor:
    """
    An interface to framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, data: Optional[TTensor]):
        self._data = data.data if isinstance(data, Tensor) else data

    @property
    def data(self) -> TTensor:
        return self._data

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def ndim(self) -> int:
        return cast(int, self.data.ndim)

    @property
    def device(self) -> TensorDeviceType:
        return cast(TensorDeviceType, _call_function("device", self))

    @property
    def dtype(self) -> TensorDataType:
        return cast(TensorDataType, _call_function("dtype", self))

    @property
    def backend(self) -> TensorBackend:
        return cast(TensorBackend, _call_function("backend", self))

    @property
    def size(self) -> int:
        return cast(int, _call_function("size", self))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __iter__(self) -> Iterator[Tensor]:
        return TensorIterator(self)

    def __getitem__(self, index: Union[Tensor, int, tuple[Union[Tensor, int], ...]]) -> Tensor:
        return Tensor(self.data[unwrap_index(index)])

    def __setitem__(self, index: Union[Tensor, int, tuple[Union[Tensor, int], ...]], value: Any) -> None:
        self.data[unwrap_index(index)] = unwrap_tensor_data(value)

    def __str__(self) -> str:
        return f"nncf.Tensor({str(self.data)})"

    def __repr__(self) -> str:
        return f"nncf.Tensor({repr(self.data)})"

    def __len__(self) -> int:
        return len(self.data)

    # built-in operations

    def __or__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data | unwrap_tensor_data(other))

    def __and__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data & unwrap_tensor_data(other))

    def __add__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data + unwrap_tensor_data(other))

    def __radd__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) + self.data)

    def __iadd__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        self._data += unwrap_tensor_data(other)
        return self

    def __sub__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data - unwrap_tensor_data(other))

    def __rsub__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) - self.data)

    def __isub__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        self._data -= unwrap_tensor_data(other)
        return self

    def __mul__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data * unwrap_tensor_data(other))

    def __rmul__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) * self.data)

    def __imul__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        self._data *= unwrap_tensor_data(other)
        return self

    def __pow__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data ** unwrap_tensor_data(other))

    def __rpow__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(unwrap_tensor_data(other) ** self.data)

    def __ipow__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        self._data **= unwrap_tensor_data(other)
        return self

    def __truediv__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return cast(Tensor, _call_function("_binary_op_nowarn", self, other, operator.truediv))

    def __rtruediv__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return cast(Tensor, _call_function("_binary_reverse_op_nowarn", self, other, operator.truediv))

    def __itruediv__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        self._data /= unwrap_tensor_data(other)
        return self

    def __floordiv__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return cast(Tensor, _call_function("_binary_op_nowarn", self, other, operator.floordiv))

    def __rfloordiv__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return cast(Tensor, _call_function("_binary_reverse_op_nowarn", self, other, operator.floordiv))

    def __ifloordiv__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        self._data //= unwrap_tensor_data(other)
        return self

    def __mod__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return cast(Tensor, _call_function("_binary_op_nowarn", self, other, operator.mod))

    def __matmul__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data @ unwrap_tensor_data(other))

    def __neg__(self) -> Tensor:
        return Tensor(-self.data)

    def __invert__(self) -> Tensor:
        return Tensor(~self.data)

    def __rshift__(self, other: T_NUMBER) -> Tensor:
        return Tensor(self.data >> unwrap_tensor_data(other))

    def __lshift__(self, other: T_NUMBER) -> Tensor:
        return Tensor(self.data << unwrap_tensor_data(other))

    # Comparison operators

    def __lt__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data < unwrap_tensor_data(other))

    def __le__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data <= unwrap_tensor_data(other))

    def __eq__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:  # type: ignore[override]
        return Tensor(self.data == unwrap_tensor_data(other))

    def __ne__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:  # type: ignore[override]
        return Tensor(self.data != unwrap_tensor_data(other))

    def __gt__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data > unwrap_tensor_data(other))

    def __ge__(self, other: Union[Tensor, T_NUMBER]) -> Tensor:
        return Tensor(self.data >= unwrap_tensor_data(other))

    # Tensor functions

    def squeeze(self, axis: T_AXIS = None) -> Tensor:
        return cast(Tensor, _call_function("squeeze", self, axis))

    def flatten(self) -> Tensor:
        return cast(Tensor, _call_function("flatten", self))

    def max(self, axis: T_AXIS = None, keepdims: Optional[bool] = False) -> Tensor:
        return cast(Tensor, _call_function("max", self, axis, keepdims))

    def min(self, axis: T_AXIS = None, keepdims: Optional[bool] = False) -> Tensor:
        return cast(Tensor, _call_function("min", self, axis, keepdims))

    def abs(self) -> Tensor:
        return cast(Tensor, _call_function("abs", self))

    def isempty(self) -> bool:
        return cast(bool, _call_function("isempty", self))

    def astype(self, dtype: TensorDataType) -> Tensor:
        return cast(Tensor, _call_function("astype", self, dtype))

    def view(self, dtype: TensorDataType) -> Tensor:
        return cast(Tensor, _call_function("view", self, dtype))

    def reshape(self, shape: T_SHAPE) -> Tensor:
        return cast(Tensor, _call_function("reshape", self, shape))

    def item(self) -> T_NUMBER:
        return cast(T_NUMBER, _call_function("item", self))

    def clone(self) -> Tensor:
        return cast(Tensor, _call_function("clone", self))

    def as_numpy_tensor(self) -> Tensor:
        return cast(Tensor, _call_function("as_numpy_tensor", self))

    def tolist(self) -> Any:
        return _call_function("tolist", self)

    def as_openvino_tensor(self) -> Tensor:
        x = self
        if x.backend == TensorBackend.numpy:
            x = cast(Tensor, _call_function("from_numpy", x.data, backend=TensorBackend.ov))
        if x.backend != TensorBackend.ov:
            msg = f"Unsupported backend for OpenVINO conversion: {x.backend}."
            raise NotImplementedError(msg)
        return x


def _call_function(func_name: str, *args: Any, **kwargs: Any) -> Any:
    """
    Call function from functions.py to avoid circular imports.

    :param func_name: Name of function.
    :return: Result of function call.
    """
    from nncf.tensor.functions import numeric

    fn = getattr(numeric, func_name)
    return fn(*args, **kwargs)


class TensorIterator(Iterator[Tensor]):
    """Iterator for Tensor class"""

    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor
        self._index = 0

    def __next__(self) -> Tensor:
        tensor_shape = self._tensor.shape
        if not tensor_shape:
            msg = "iteration over a 0-d tensor"
            raise TypeError(msg)
        if self._index < tensor_shape[0]:
            result = self._tensor[self._index]
            self._index += 1
            return result

        raise StopIteration


def unwrap_index(obj: Union[Any, tuple[Any, ...]]) -> Union[TTensor, tuple[TTensor, ...]]:
    """
    Unwraps the tensor data from the input object or tuple of objects.

    :param obj: The object to unwrap.
    :return: The unwrapped tensor data or tuple of unwrapped tensor data.
    """
    if isinstance(obj, tuple):
        return tuple(unwrap_tensor_data(o) for o in obj)
    return unwrap_tensor_data(obj)


def unwrap_tensor_data(obj: Any) -> TTensor:
    """
    Return the data of a Tensor object, or the object itself if it is not a Tensor.

    :param obj: The object to unwrap.
    :return: The data of the Tensor object, or the object itself.
    """
    return obj.data if isinstance(obj, Tensor) else obj


def get_tensor_backend(backend: BackendType) -> TensorBackend:
    """
    Returns a tensor backend based on the provided backend.

    :param backend: Backend type.
    :return: Corresponding tensor backend type.
    """
    BACKEND_TO_TENSOR_BACKEND: dict[BackendType, TensorBackend] = {
        BackendType.OPENVINO: TensorBackend.numpy,
        BackendType.ONNX: TensorBackend.numpy,
        BackendType.TORCH_FX: TensorBackend.torch,
        BackendType.TORCH: TensorBackend.torch,
    }
    if backend not in BACKEND_TO_TENSOR_BACKEND:
        msg = f"Unsupported backend type: {backend}"
        raise nncf.ValidationError(msg)

    return BACKEND_TO_TENSOR_BACKEND[backend]
