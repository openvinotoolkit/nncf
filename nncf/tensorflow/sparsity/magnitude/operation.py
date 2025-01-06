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

import tensorflow as tf

from nncf.tensorflow.graph.utils import get_weight_by_name
from nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.sparsity.magnitude.functions import apply_mask


@NNCF_CUSTOM_OBJECTS.register()
class BinaryMask(NNCFOperation):
    def build(self, input_shape, input_type, name, layer):
        if input_type is not InputType.WEIGHTS:
            raise ValueError("Binary Mask operation could not be applied to input of the layer: {}".format(layer.name))

        mask = layer.add_weight(
            name + "_mask",
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
            aggregation=tf.VariableAggregation.MEAN,
        )

        return {"mask": mask}

    def call(self, inputs, weights, _):
        return apply_mask(inputs, weights["mask"])

    @staticmethod
    def get_binary_mask(op_weights):
        """
        Returns binary mask from weights of the operation.

        :param op_weights: Weights of the operaton.
        :return: Binary mask.
        """
        return op_weights["mask"]


@NNCF_CUSTOM_OBJECTS.register()
class BinaryMaskWithWeightsBackup(BinaryMask):
    def __init__(self, name: str, w_name_to_bkup: str = None):
        super().__init__(name)
        self.w_name_to_bkup = w_name_to_bkup
        self.bkup_var = None

    def build(self, input_shape, input_type, name, layer):
        self.bkup_var = self._create_bkup_weights(layer, self.w_name_to_bkup)
        return super().build(input_shape, input_type, name, layer)

    def call(self, inputs, weights, _):
        self.bkup_var.assign(tf.where(weights["mask"] > 0.5, inputs, self.bkup_var))
        return apply_mask(self.bkup_var, weights["mask"])

    @staticmethod
    def _create_bkup_weights(layer, w_name):
        var = get_weight_by_name(layer, w_name)
        bkup_var = layer.add_weight(
            w_name + "_bkup", shape=var.shape, trainable=False, aggregation=tf.VariableAggregation.MEAN
        )

        bkup_var.assign(var.read_value())
        return bkup_var

    def get_config(self):
        config = super().get_config()
        config["w_name_to_bkup"] = self.w_name_to_bkup
        return config
