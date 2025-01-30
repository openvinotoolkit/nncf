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

from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.quantization.layers import FakeQuantize


def apply_overflow_fix(model: tf.keras.Model, op_names: List[str]) -> None:
    if not isinstance(model, tf.keras.Model):
        raise ValueError(f"Expected model to be a `tf.keras.Model` instance but got: {type(model)}")

    for wrapped_layer, weight_attr, op in get_nncf_operations(model, op_names):
        if op.half_range:
            apply_overflow_fix_to_layer(wrapped_layer, weight_attr, op)


def apply_overflow_fix_to_layer(wrapped_layer: NNCFWrapper, weight_attr: str, op: NNCFOperation) -> None:
    layer_weight = wrapped_layer.layer_weights[weight_attr]
    ops_weights = wrapped_layer.get_operation_weights(op.name)
    # Keep zero weights to prevent
    # zero quant calculation arithmetic errors
    mask = layer_weight == 0.0
    layer_weight_updated = op.call(layer_weight, ops_weights, False)

    # Assign exact zero to weights which
    # was exact zero before overflow fix
    layer_weight_updated = tf.where(mask, [0.0], layer_weight_updated)
    layer_weight.assign(layer_weight_updated)
    op.apply_overflow_fix(ops_weights)


def collect_fake_quantize_layers(model: tf.keras.Model) -> List[FakeQuantize]:
    """
    Collects all fake quantize layers from the provided model.

    :param model: An instance of the `tf.keras.Model` class.
    :return: A list of fake quantize layers.
    """
    fq_layers = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            fq_layers.extend(collect_fake_quantize_layers(layer))
        if isinstance(layer, FakeQuantize):
            fq_layers.append(layer)
    return fq_layers
