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

from typing import List

import tensorflow as tf

from nncf.tensorflow.graph.model_transformer import TFModelTransformer
from nncf.tensorflow.graph.transformations.commands import TFOperationWithWeights
from nncf.tensorflow.graph.transformations.commands import TFRemovalCommand
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.layers.wrapper import NNCFWrapper


def strip_model_from_masks(model: tf.keras.Model, op_names: List[str]) -> tf.keras.Model:
    if not isinstance(model, tf.keras.Model):
        raise ValueError(f"Expected model to be a `tf.keras.Model` instance but got: {type(model)}")

    transformations = TFTransformationLayout()
    for wrapped_layer, weight_attr, op in get_nncf_operations(model, op_names):
        apply_mask(wrapped_layer, weight_attr, op)
        transformations.register(
            TFRemovalCommand(
                target_point=TFOperationWithWeights(
                    wrapped_layer.name, weights_attr_name=weight_attr, operation_name=op.name
                )
            )
        )

    return TFModelTransformer(model).transform(transformations)


def apply_mask(wrapped_layer: NNCFWrapper, weight_attr: str, op: NNCFOperation) -> None:
    layer_weight = wrapped_layer.layer_weights[weight_attr]
    op_weights = wrapped_layer.get_operation_weights(op.name)
    layer_weight.assign(op(layer_weight, op_weights, False))
    wrapped_layer.set_layer_weight(weight_attr, layer_weight)
