"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns import merge_two_types_of_operations

LINEAR_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [
        "linear",
        "conv1d",
        "conv2d",
        "conv3d",
        "conv_transpose1d",
        "conv_transpose2d",
        "conv_transpose3d",
        "deform_conv2d",
        "addmm",
        "bmm",
        "matmul",
        "mm",
        "baddbmm",
    ],
    GraphPattern.LABEL_ATTR: "LINEAR",
}

BATCH_NORMALIZATION_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: ["batch_norm", "batch_norm1d", "batch_norm2d", "batch_norm3d"],
    GraphPattern.LABEL_ATTR: "BATCH_NORMALIZATION",
}

GROUP_NORMALIZATION_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: ["group_norm"],
    GraphPattern.LABEL_ATTR: "GROUP_NORMALIZATION",
}

LAYER_NORMALIZATION_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: ["layer_norm"],
    GraphPattern.LABEL_ATTR: "LAYER_NORMALIZATION",
}

RELU_OPERATIONS = {GraphPattern.METATYPE_ATTR: ["relu", "relu_", "hardtanh"], GraphPattern.LABEL_ATTR: "RELU"}

NON_RELU_ACTIVATIONS_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: [
        "elu",
        "elu_",
        "prelu",
        "leaky_relu",
        "sigmoid",
        "gelu",
        "silu",
        "hardsigmoid",
        "hardswish",
    ],
    GraphPattern.LABEL_ATTR: "NON_RELU_ACTIVATIONS",
}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(
    RELU_OPERATIONS, NON_RELU_ACTIVATIONS_OPERATIONS, "ATOMIC_ACTIVATIONS"
)

ARITHMETIC_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: ["__iadd__", "__add__", "__mul__", "__rmul__", "__truediv__"],
    GraphPattern.LABEL_ATTR: "ARITHMETIC",
}

# This type may be useful in the future
# pylint: disable=unused-variable
POOLING_OPERATIONS = {
    GraphPattern.METATYPE_ATTR: ["adaptive_avg_pool2d", "adaptive_avg_pool3d", "avg_pool2d", "avg_pool3d"],
    GraphPattern.LABEL_ATTR: "POOLING",
}
