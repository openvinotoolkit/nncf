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
    AVGPOOL = auto()
    BROADCAST = auto()
    CHUNK = auto()
    CONCAT = auto()
    CONVERTLIKE = auto()
    CONVOLUTION = auto()
    CROP = auto()
    DEPTHWISECONVOLUTION = auto()
    DIVIDE = auto()
    EMBEDDING = auto()
    EMBEDDINGBAG = auto()
    EQUAL = auto()
    FLATTEN = auto()
    FLOORMOD = auto()
    GELU = auto()
    GREATER = auto()
    GREATEREQUAL = auto()
    GROUPNORMALIZATION = auto()
    GRUSEQUENCE = auto()
    INTERPOLATE = auto()
    LESS = auto()
    LESSEQUAL = auto()
    LOGICALAND = auto()
    LOGICALNOT = auto()
    LOGICALOR = auto()
    LOGICALXOR = auto()
    LSTMSEQUENCE = auto()
    MATMUL = auto()
    MAXIMUM = auto()
    MAXPOOL = auto()
    MINIMUM = auto()
    MULTIPLY = auto()
    MVN = auto()
    NORMALIZEL2 = auto()
    NOTEQUAL = auto()
    PAD = auto()
    POWER = auto()
    REDUCEL2 = auto()
    REDUCEMAX = auto()
    REDUCEMEAN = auto()
    REDUCESUM = auto()
    RESHAPE = auto()
    SCALED_DOT_PRODUCT_ATTENTION = auto()
    SHUFFLECHANNELS = auto()
    SLICE = auto()
    SPLIT = auto()
    SQUEEZE = auto()
    STRIDEDSLICE = auto()
    SUBTRACT = auto()
    TILE = auto()
    TRANSPOSE = auto()
    UNSQUEEZE = auto()
    VARIADICSPLIT = auto()
