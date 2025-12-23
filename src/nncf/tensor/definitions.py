# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Optional, Union

from nncf.parameters import StrEnum

T_SHAPE_ARRAY = tuple[int, ...]
T_SHAPE = Union[int, T_SHAPE_ARRAY]
T_AXIS = Optional[T_SHAPE]
T_NUMBER = Union[int, float, bool]


class TensorBackend(Enum):
    """
    Enum representing the different tensor backends.
    """

    numpy = auto()
    torch = auto()
    ov = auto()


class TensorDataType(StrEnum):
    """
    Enum representing the different tensor data types.
    """

    float16 = auto()
    bfloat16 = auto()
    float32 = auto()
    float64 = auto()
    f8e4m3 = auto()
    f8e5m2 = auto()
    f8e8m0 = auto()
    f4e2m1 = auto()
    nf4 = auto()
    int8 = auto()
    int32 = auto()
    int64 = auto()
    uint16 = auto()
    uint32 = auto()
    uint8 = auto()
    uint4 = auto()
    int4 = auto()

    def is_float(self) -> bool:
        """
        :return: True if the tensor data type is a floating-point type, else False.
        """
        return self in [
            TensorDataType.float16,
            TensorDataType.bfloat16,
            TensorDataType.float32,
            TensorDataType.float64,
            TensorDataType.f8e4m3,
            TensorDataType.f8e5m2,
            TensorDataType.nf4,
        ]

    def itemsize(self) -> int:
        """
        Returns the size of a single item in bits for the tensor data type.

        :return: The item size in bits.
        """
        itemsize_bits = {
            TensorDataType.nf4: 4,
            TensorDataType.uint4: 4,
            TensorDataType.int4: 4,
            TensorDataType.f8e4m3: 8,
            TensorDataType.f8e5m2: 8,
            TensorDataType.int8: 8,
            TensorDataType.uint8: 8,
            TensorDataType.uint16: 16,
            TensorDataType.uint32: 32,
            TensorDataType.float16: 16,
            TensorDataType.bfloat16: 16,
            TensorDataType.float32: 32,
            TensorDataType.int32: 32,
            TensorDataType.float64: 64,
            TensorDataType.int64: 64,
        }

        return itemsize_bits[self]


class TensorDeviceType(Enum):
    """
    Enum representing the different tensor device types.
    """

    CPU = auto()
    GPU = auto()


@dataclass
class TypeInfo:
    """
    The class represents the numerical properties of a floating point types.

    :param eps: The smallest representable number such that 1.0 + eps != 1.0.
    :param max: The largest representable number.
    :param min: The smallest representable number (typically -max).
    """

    eps: float
    max: float
    min: float
