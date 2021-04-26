"""
 Copyright (c) 2021 Intel Corporation
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

from nncf.tensorflow.graph.utils import get_nncf_operations
from nncf.tensorflow.layers.wrapper import NNCFWrapper


def apply_saturation_fix(model: tf.keras.Model, op_names: List[str]) -> None:
    if not isinstance(model, tf.keras.Model):
        raise ValueError(f'Expected model to be a `tf.keras.Model` instance but got: {type(model)}')

    for wrapped_layer, weight_attr, op in get_nncf_operations(model, op_names):
        if op.half_range:
            apply_saturation_fix_to_layer(wrapped_layer, weight_attr, op.name)


def apply_saturation_fix_to_layer(wrapped_layer: NNCFWrapper, weight_attr: str, op_name: str) -> None:
    layer_weight = wrapped_layer.layer_weights[weight_attr]
    op = wrapped_layer.weights_attr_ops[weight_attr][op_name]
    ops_weights = wrapped_layer.ops_weights[op_name]
    layer_weight.assign(
        op.call(layer_weight, ops_weights, False)
    )
    op.apply_saturation_fix(ops_weights)
