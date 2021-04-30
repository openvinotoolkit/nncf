"""
 Copyright (c) 2020 Intel Corporation
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

from typing import List

import tensorflow as tf

from beta.nncf.tensorflow.graph.model_transformer import TFModelTransformer
from beta.nncf.tensorflow.graph.transformations.commands import TFOperationWithWeights
from beta.nncf.tensorflow.graph.transformations.commands import TFRemovalCommand
from beta.nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from beta.nncf.tensorflow.graph.utils import collect_wrapped_layers
from beta.nncf.tensorflow.layers.wrapper import NNCFWrapper


def strip_model_from_masks(model: tf.keras.Model, op_names: List[str]) -> tf.keras.Model:
    if not isinstance(model, tf.keras.Model):
        raise ValueError(
            'Expected model to be a `tf.keras.Model` instance but got: {}'.format(type(model)))

    transformations = TFTransformationLayout()

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            for weight_attr, ops in layer.weights_attr_ops.items():
                for op_name in ops:
                    if op_name in op_names:
                        apply_mask(layer, weight_attr, op_name)

                        transformations.register(
                            TFRemovalCommand(
                                target_point=TFOperationWithWeights(
                                    layer.name,
                                    weights_attr_name=weight_attr,
                                    operation_name=op_name)
                            ))

    return TFModelTransformer(model, transformations).transform()


def apply_fn_to_op_weights(model: tf.keras.Model, op_names: List[str], fn = lambda x: x):
    sparsifyed_layers = collect_wrapped_layers(model)
    target_ops = []
    for layer in sparsifyed_layers:
        for ops in layer.weights_attr_ops.values():
            for op in ops.values():
                if op.name in op_names:
                    weight = layer.get_operation_weights(op.name)
                    target_ops.append((op, fn(weight)))
    return target_ops


def apply_mask(wrapped_layer: NNCFWrapper, weight_attr: str, op_name: str):
    layer_weight = wrapped_layer.layer_weights[weight_attr]
    op = wrapped_layer.weights_attr_ops[weight_attr][op_name]
    layer_weight.assign(
        op(layer_weight,
           wrapped_layer.ops_weights[op_name],
           False)
    )
    wrapped_layer.set_layer_weight(weight_attr, layer_weight)
