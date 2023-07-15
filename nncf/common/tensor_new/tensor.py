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

from nncf.common.tensor_new import numpy_ops
from nncf.common.tensor_new.enums import TensorBackendType
from nncf.common.tensor_new.enums import TensorDataType

try:
    from nncf.torch import torch_ops

except ImportError:
    torch_ops = None


DataType = TypeVar("DataType")
DeviceType = TypeVar("DeviceType")


FUNC_MAP_DISPATCHER = {
    TensorBackendType.NUMPY: numpy_ops,
}
if torch_ops:
    FUNC_MAP_DISPATCHER[TensorBackendType.TORCH] = torch_ops


class Tensor:
    """
    An interface to framework specific tensors for common NNCF algorithms.
    """

    def __init__(self, data: Optional[DataType]):
        self._data = data.data if isinstance(data, Tensor) else data

    @property
    def data(self) -> DataType:
        return self._data

    @property
    def shape(self) -> List[int]:
        return Tensor(list(self.data.shape))

    def __bool__(self) -> bool:
        return bool(self.data)

    def __iter__(self) -> Iterator:
        return iter(self.data)

    def __getitem__(self, index: int) -> "Tensor":
        return Tensor(self.data[index])

    # built-in operations

    def __add__(self, other: DataType) -> "Tensor":
        return Tensor(self.data + unwrap_tensor_data(other))

    def __radd__(self, other: DataType) -> "Tensor":
        return Tensor(unwrap_tensor_data(other) + self.data)

    def __sub__(self, other: DataType) -> "Tensor":
        return Tensor(self.data - unwrap_tensor_data(other))

    def __rsub__(self, other: DataType) -> "Tensor":
        return Tensor(unwrap_tensor_data(other) - self.data)

    def __mul__(self, other: DataType) -> "Tensor":
        return Tensor(self.data * unwrap_tensor_data(other))

    def __rmul__(self, other: DataType) -> "Tensor":
        return Tensor(unwrap_tensor_data(other) * self.data)

    def __pow__(self, other: DataType) -> "Tensor":
        return Tensor(self.data ** unwrap_tensor_data(other))

    def __truediv__(self, other: DataType) -> "Tensor":
        return Tensor(self.data / unwrap_tensor_data(other))

    def __floordiv__(self, other: DataType) -> "Tensor":
        return Tensor(self.data // unwrap_tensor_data(other))

    def __neg__(self) -> "Tensor":
        return Tensor(-self.data)

    # Comparison operators

    def __lt__(self, other: DataType) -> "Tensor":
        return Tensor(self.data < unwrap_tensor_data(other))

    def __le__(self, other: DataType) -> "Tensor":
        return Tensor(self.data <= unwrap_tensor_data(other))

    def __eq__(self, other: DataType) -> "Tensor":
        return Tensor(self.data == unwrap_tensor_data(other))

    def __nq__(self, other: DataType) -> "Tensor":
        return Tensor(self.data != unwrap_tensor_data(other))

    def __gt__(self, other: DataType) -> "Tensor":
        return Tensor(self.data > unwrap_tensor_data(other))

    def __ge__(self, other: DataType) -> "Tensor":
        return Tensor(self.data >= unwrap_tensor_data(other))

    # Tensor functions

    @property
    def device(self) -> Optional[DeviceType]:
        return tensor_func_dispatcher("device", self.data)

    def squeeze(self, axis: Optional[Union[int, Tuple[int]]] = None) -> "Tensor":
        return tensor_func_dispatcher("squeeze", self.data, axis=axis)

    def flatten(self) -> "Tensor":
        return tensor_func_dispatcher("flatten", self.data)

    def max(self, axis: Optional[DataType] = None) -> "Tensor":
        return tensor_func_dispatcher("max", self.data, axis=axis)

    def min(self, axis: Optional[DataType] = None) -> "Tensor":
        return tensor_func_dispatcher("min", self.data, axis=axis)

    def abs(self) -> "Tensor":
        return tensor_func_dispatcher("absolute", self.data)

    def is_empty(self) -> "Tensor":
        return tensor_func_dispatcher("is_empty", self.data)

    def as_type(self, dtype: TensorDataType):
        return tensor_func_dispatcher("as_type", self.data, dtype)

    def reshape(self, shape: DataType) -> "Tensor":
        return tensor_func_dispatcher("reshape", self.data, shape)


def unwrap_tensor_data(obj: Any) -> DataType:
    """
    Return the data of a Tensor object, or the object itself if it is not a Tensor.

    :param obj: The object to unwrap.
    :return: The data of the Tensor object, or the object itself.
    """
    return obj.data if isinstance(obj, Tensor) else obj


def detect_tensor_backend(*args):
    """
    Detect the backend of a Tensor object.

    :param *args: The arguments to the Tensor object.
    :return: The backend of the Tensor object, or None if the backend cannot be determined.
    """
    if not args:
        raise RuntimeError("tensor_func_dispatcher detect backend by args[0]")

    for backend, tensor_ops in FUNC_MAP_DISPATCHER.items():
        if tensor_ops.check_tensor_backend(args[0]):
            return backend
    return None


def tensor_func_dispatcher(func_name: str, *args, **kwargs) -> Any:
    """
    Dispatcher for tensor functions.

    :param func_name: The name of the function to dispatch.
    :param *args: The arguments to the function.
    :param **kwargs: The keyword arguments to the function.
    :return: The result of the function call.
    """
    args = tuple(map(unwrap_tensor_data, args))
    kwargs = {k: unwrap_tensor_data(v) for k, v in kwargs.items()}

    tensor_backend = detect_tensor_backend(*args)
    if tensor_backend is None:
        raise RuntimeError(f"{func_name} is not implemented for {type(args[0])}")

    module_ops = FUNC_MAP_DISPATCHER[tensor_backend]
    func = getattr(module_ops, func_name)

    return Tensor(func(*args, **kwargs))
