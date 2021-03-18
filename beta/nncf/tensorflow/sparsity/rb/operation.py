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

from beta.nncf.tensorflow.functions import logit
from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.operation import InputType
from beta.nncf.tensorflow.layers.operation import NNCFOperation
from beta.nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from beta.nncf.tensorflow.sparsity.rb.functions import calc_rb_binary_mask, st_binary_mask, binary_mask

OP_NAME = 'rb_sparsity_mask_apply'

@NNCF_CUSTOM_OBJECTS.register()
class RBSparsifyingWeight(NNCFOperation):

    def __init__(self, eps=1e-6):
        '''
        :param eps: minimum value and the gap from the maximum value in
            distributed mask
        '''
        super().__init__(name=OP_NAME)
        self.eps = eps

    # TODO: make it static
    def build(self, input_shape, input_type, name, layer):
        '''
        :param input_shape: shape of weights which needs to be sparsifyed
        :param input_type: type of operation input, must be InputType.WEIGHTS
        :param name: name of layer which needs to be sparsifyed
        :param layer: layer which needs to be sparsifyed
        '''
        if input_type is not InputType.WEIGHTS:
            raise ValueError(
                'RB Sparsity mask operation could not be applied to input of the layer: {}'.
                    format(layer.name))

        mask = layer.add_weight(
            name + '_mask',
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(logit(0.99)),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN)

        trainable = layer.add_weight(
            name + '_trainable',
            initializer=tf.keras.initializers.Constant(True),
            trainable=False,
            dtype=tf.bool)

        return {
            'mask': mask,
            'trainable': trainable,
        }

    def call(self, layer_weights, op_weights, trainable):
        '''
        Apply rb sparsity mask to given weights

        :param layer_weights: target weights to sparsify
        :param op_weights: operation weights contains
           mask and param `trainable`
        :param trainable: true if operation called in training mode
        '''
        return tf.cond(op_weights['trainable'],
                       true_fn=lambda: apply_mask(layer_weights, calc_rb_binary_mask(op_weights['mask'], self.eps)),
                       false_fn=lambda: apply_mask(layer_weights, binary_mask(op_weights['mask'])))

    def freeze(self, op_weights):
        '''
        Freeze rb mask from operation weights

        :param op_weights: weight of rb operation
        '''
        op_weights['trainable'].assign(False)

    @staticmethod
    def loss(mask):
        '''
        Return count of non zero weight in mask

        :param mask: given mask
        '''
        return tf.reduce_sum(st_binary_mask(mask))
