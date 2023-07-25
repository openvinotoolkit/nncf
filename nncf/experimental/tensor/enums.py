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

from enum import Enum
from enum import auto


class TensorBackendType(Enum):
    """
    Enum representing the different tensor backends.
    """

    NUMPY = auto()
    TORCH = auto()


class TensorDataType(Enum):
    """
    Enum representing the different tensor data types.
    """

    float16 = auto()
    float32 = auto()
    float64 = auto()
    int8 = auto()
    uint8 = auto()


class TensorDeviceType(Enum):
    """
    Enum representing the different tensor device types.
    """

    CPU = auto()
    GPU = auto()
