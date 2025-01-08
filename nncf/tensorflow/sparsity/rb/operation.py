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

import numpy as np
import tensorflow as tf

from nncf.tensorflow import tf_internals
from nncf.tensorflow.functions import logit
from nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.layers.wrapper import NNCFWrapper
from nncf.tensorflow.sparsity.magnitude.functions import apply_mask
from nncf.tensorflow.sparsity.rb.functions import binary_mask
from nncf.tensorflow.sparsity.rb.functions import calc_rb_binary_mask
from nncf.tensorflow.sparsity.rb.functions import st_binary_mask


@NNCF_CUSTOM_OBJECTS.register()
class RBSparsifyingWeight(NNCFOperation):
    def __init__(self, name: str, eps: float = 1e-6):
        """
        :param name: Model scope unique operation name.
        :param eps: Minimum value and the gap from the maximum value
            in the mask.
        """
        super().__init__(name)
        self.eps = eps

    def build(self, input_shape, input_type: InputType, name: str, layer: NNCFWrapper):
        """
        :param input_shape: Shape of weights which needs to be sparsifyed.
        :param input_type: Type of operation input, must be InputType.WEIGHTS.
        :param name: Name of weight attribute which needs to be sparsifyed.
        :param layer: Layer which needs to be sparsifyed.
        """
        if input_type is not InputType.WEIGHTS:
            raise ValueError(
                "RB Sparsity mask operation could not be applied to input of the layer: {}".format(layer.name)
            )

        mask = layer.add_weight(
            name + "_mask",
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(logit(0.99)),
            trainable=True,
            aggregation=tf.VariableAggregation.MEAN,
        )

        trainable = layer.add_weight(
            name + "_trainable", initializer=tf.keras.initializers.Constant(True), trainable=False, dtype=tf.bool
        )

        seed = layer.add_weight(
            name + "_seed",
            shape=(2,),
            initializer=tf.keras.initializers.Constant(np.random.randint(size=(2,), low=-(2**31), high=2**31 - 1)),
            trainable=False,
            dtype=tf.int32,
        )

        return {
            "mask": mask,
            "trainable": trainable,
            "seed": seed,
        }

    def call(self, inputs, weights, training: tf.constant):
        """
        Apply rb sparsity mask to given weights.

        :param inputs: Target weights to sparsify.
        :param weights: Operation weights contains
            `mask` and param `trainable`.
        :param training: True if operation called in training mode
            else False
        """
        true_fn = lambda: apply_mask(inputs, self._calc_rb_binary_mask(weights))
        false_fn = lambda: apply_mask(inputs, binary_mask(weights["mask"]))
        return tf_internals.smart_cond(
            training,
            true_fn=lambda: tf_internals.smart_cond(weights["trainable"], true_fn=true_fn, false_fn=false_fn),
            false_fn=false_fn,
        )

    def _calc_rb_binary_mask(self, op_weights):
        new_seed = tf.random.stateless_uniform((2,), seed=op_weights["seed"], minval=-(2**31), maxval=2**31 - 1)
        new_seed = tf.cast(new_seed, tf.int32)
        op_weights["seed"].assign(new_seed)
        return calc_rb_binary_mask(op_weights["mask"], op_weights["seed"], self.eps)

    def freeze(self, op_weights):
        """
        Freeze rb mask from operation weights.

        :param op_weights: Operation weights.
        """
        op_weights["trainable"].assign(False)

    @staticmethod
    def loss(op_weights):
        """
        Return count of non zero weight in mask.

        :param op_weights: Operation weights.
        """
        return tf.reduce_sum(st_binary_mask(op_weights["mask"]))

    @staticmethod
    def get_mask(op_weights):
        """
        Return mask weight from operation weights.

        :param op_weights: Operation weights.
        """
        return op_weights["mask"]

    @staticmethod
    def get_binary_mask(op_weights):
        """
        Returns binary mask from weights of the operation.

        :param op_weights: Weights of the operaton.
        :return: Binary mask.
        """
        return binary_mask(op_weights["mask"])

    @staticmethod
    def get_trainable_weight(op_weights):
        """
        Return trainable weight from operation weights.

        :param op_weights: Operation weights.
        """
        return op_weights["trainable"]

    def get_config(self):
        config = super().get_config()
        config["eps"] = self.eps
        return config
