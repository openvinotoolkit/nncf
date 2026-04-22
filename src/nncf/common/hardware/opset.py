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

from enum import auto

from nncf.parameters import StrEnum


class HWConfigOpName(StrEnum):
    """
    Enumeration of operation types supported by hardware quantization configurations.

    Each member represents an operation type that can be quantized on a specific hardware platform
    (CPU, GPU, NPU). These types are used to define quantization schemes and constraints in
    hardware-specific setup configurations.
    """

    ADD = auto()
    AVG_POOL = auto()
    BROADCAST = auto()
    CHUNK = auto()
    CONCAT = auto()
    CONVERT_LIKE = auto()
    CONVOLUTION = auto()
    CROP = auto()
    DEPTHWISE_CONVOLUTION = auto()
    DIVIDE = auto()
    EMBEDDING = auto()
    EMBEDDING_BAG = auto()
    EQUAL = auto()
    FLATTEN = auto()
    FLOOR_MOD = auto()
    GELU = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    GROUP_NORMALIZATION = auto()
    GRU_SEQUENCE = auto()
    INTERPOLATE = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    LOGICAL_AND = auto()
    LOGICAL_NOT = auto()
    LOGICAL_OR = auto()
    LOGICAL_XOR = auto()
    LSTM_SEQUENCE = auto()
    MAT_MUL = auto()
    MAXIMUM = auto()
    MAX_POOL = auto()
    MINIMUM = auto()
    MULTIPLY = auto()
    MVN = auto()
    NORMALIZE_L2 = auto()
    NOT_EQUAL = auto()
    PAD = auto()
    POWER = auto()
    REDUCE_L2 = auto()
    REDUCE_MAX = auto()
    REDUCE_MEAN = auto()
    REDUCE_SUM = auto()
    RESHAPE = auto()
    SCALED_DOT_PRODUCT_ATTENTION = auto()
    SHUFFLE_CHANNELS = auto()
    SLICE = auto()
    SPLIT = auto()
    SQUEEZE = auto()
    STRIDED_SLICE = auto()
    SUBTRACT = auto()
    TILE = auto()
    TRANSPOSE = auto()
    UNSQUEEZE = auto()
    VARIADIC_SPLIT = auto()
