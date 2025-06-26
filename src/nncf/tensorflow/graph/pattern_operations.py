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
from nncf.common.graph.patterns import merge_two_types_of_operations
from nncf.tensorflow.graph.metatypes.common import ELEMENTWISE_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import GENERAL_CONV_LAYER_METATYPES
from nncf.tensorflow.graph.metatypes.common import LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT
from nncf.tensorflow.graph.metatypes.common import LINEAR_LAYER_METATYPES

LINEAR_OPERATIONS = {
    "type": list(
        {
            *{layer_name for m in GENERAL_CONV_LAYER_METATYPES for layer_name in m.get_all_aliases()},
            *{layer_name for m in LINEAR_LAYER_METATYPES for layer_name in m.get_all_aliases()},
        }
    ),
    "label": "LINEAR",
}

ELEMENTWISE_OPERATIONS = {
    "type": list(set(layer_name for m in ELEMENTWISE_LAYER_METATYPES for layer_name in m.get_all_aliases())),
    "label": "ELEMENTWISE",
}

QUANTIZATION_AGNOSTIC_OPERATIONS = {
    "type": list(
        set(
            layer_name
            for m in LAYER_METATYPES_AGNOSTIC_TO_DATA_PRECISION_WITH_ONE_INPUT
            for layer_name in m.get_all_aliases()
        )
    ),
    "label": "QUANTIZATION_AGNOSTIC",
}

BATCH_NORMALIZATION_OPERATIONS = {
    "type": ["BatchNormalization", "SyncBatchNormalization", "FusedBatchNormV3"],
    "label": "BATCH_NORMALIZATION",
}

KERAS_ACTIVATIONS_OPERATIONS = {
    "type": ["ReLU", "ThresholdedReLU", "ELU", "PReLU", "LeakyReLU", "Activation"],
    "label": "KERAS_ACTIVATIONS",
}


TF_ACTIVATIONS_OPERATIONS = {
    "type": [
        "Relu",
        "nn.relu",
        "Elu",
        "LeakyRelu",
        "Relu6",
        "Selu",
        "Sigmoid",
        "Tanh",
    ],
    "label": "TF_ACTIVATIONS",
}

ATOMIC_ACTIVATIONS_OPERATIONS = merge_two_types_of_operations(
    KERAS_ACTIVATIONS_OPERATIONS, TF_ACTIVATIONS_OPERATIONS, "ATOMIC_ACTIVATIONS"
)

POOLING_OPERATIONS = {
    "type": [
        "AveragePooling2D",
        "AveragePooling3D",
        "GlobalAveragePooling2D",
        "GlobalAveragePooling3D",
        "AvgPool",
        "AvgPool3D",
        "Mean",
    ],
    "label": "POOLING",
}

SINGLE_OPS = merge_two_types_of_operations(
    POOLING_OPERATIONS, {"type": ["Average", "LayerNormalization", "UpSampling2D"]}, label="SINGLE_OPS"
)

ARITHMETIC_OPERATIONS = {"type": ["__iadd__", "__add__", "__mul__", "__rmul__"], "label": "ARITHMETIC"}
