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

from nncf.parameters import StrEnum


class HWConfigOpName(StrEnum):
    """
    Enumeration of operation types supported by hardware quantization configurations.

    Each member represents an operation type that can be quantized on a specific hardware platform
    (CPU, GPU, NPU). These types are used to define quantization schemes and constraints in
    hardware-specific setup configurations.
    """

    ADD = "Add"
    AVGPOOL = "AvgPool"
    BROADCAST = "Broadcast"
    CHUNK = "Chunk"
    CONCAT = "Concat"
    CONVERTLIKE = "ConvertLike"
    CONVOLUTION = "Convolution"
    CROP = "Crop"
    DEPTHWISECONVOLUTION = "DepthWiseConvolution"
    DIVIDE = "Divide"
    EMBEDDING = "Embedding"
    EMBEDDINGBAG = "EmbeddingBag"
    EQUAL = "Equal"
    FLATTEN = "Flatten"
    FLOORMOD = "FloorMod"
    GELU = "Gelu"
    GREATER = "Greater"
    GREATEREQUAL = "GreaterEqual"
    GROUPNORMALIZATION = "GroupNormalization"
    GRUSEQUENCE = "GRUSequence"
    INTERPOLATE = "Interpolate"
    LESS = "Less"
    LESSEQUAL = "LessEqual"
    LOGICALAND = "LogicalAnd"
    LOGICALNOT = "LogicalNot"
    LOGICALOR = "LogicalOr"
    LOGICALXOR = "LogicalXor"
    LSTMSEQUENCE = "LSTMSequence"
    MATMUL = "MatMul"
    MAXIMUM = "Maximum"
    MAXPOOL = "MaxPool"
    MINIMUM = "Minimum"
    MULTIPLY = "Multiply"
    MVN = "MVN"
    NORMALIZEL2 = "NormalizeL2"
    NOTEQUAL = "NotEqual"
    PAD = "Pad"
    POWER = "Power"
    REDUCEL2 = "ReduceL2"
    REDUCEMAX = "ReduceMax"
    REDUCEMEAN = "ReduceMean"
    REDUCESUM = "ReduceSum"
    RESHAPE = "Reshape"
    SCALED_DOT_PRODUCT_ATTENTION = "ScaledDotProductAttention"
    SHUFFLECHANNELS = "ShuffleChannels"
    SLICE = "Slice"
    SPLIT = "Split"
    SQUEEZE = "Squeeze"
    STRIDEDSLICE = "StridedSlice"
    SUBTRACT = "Subtract"
    TILE = "Tile"
    TRANSPOSE = "Transpose"
    UNSQUEEZE = "Unsqueeze"
    VARIADICSPLIT = "VariadicSplit"
