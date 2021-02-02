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

import tensorflow as tf

from beta.nncf.tensorflow.graph.utils import get_weight_by_name
from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.layers.operation import NNCFOperation
from beta.nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from beta.nncf.tensorflow.sparsity.rb.functions import calc_rb_binary_mask
from beta.nncf.tensorflow.sparsity.magnitude.operation import BinaryMask


@NNCF_CUSTOM_OBJECTS.register()
class RBSparsifyingWeight(NNCFOperation):
    def build(self, input_shape, input_type, name, layer):
        if input_type is not InputType.WEIGHTS:
            raise ValueError(
                'RB Sparsity mask operation could not be applied to input of the layer: {}'.
                    format(layer.name))

        mask = layer.add_weight(
            name + '_mask',
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN)

        return mask  # TODO: should be a dictionary

    def call(self, layer_weights, previous_mask, _):
        """Apply rb sparsity mask to given weights
        :param layer_weights: target weights to sparsify
        :param previous_mask: mask from previous training iteration
        :param _:"""
        return apply_mask(layer_weights, calc_rb_binary_mask(previous_mask))
