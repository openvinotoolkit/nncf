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

from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.layers.wrapper import NNCFWrapper


class MaskOperation(NNCFOperation):
    def build(self, input_shape, input_type, name, layer):
        mask_trainable = layer.add_weight(
            name + "_mask_trainable", shape=input_shape, initializer=tf.keras.initializers.Constant(1.0), trainable=True
        )

        mask_non_trainable = layer.add_weight(
            name + "_mask_non_trainable",
            shape=input_shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=False,
        )

        return {"mask_trainable": mask_trainable, "mask_non_trainable": mask_non_trainable}

    def call(self, inputs, weights, _):
        return self.apply_mask(inputs, weights["mask_trainable"], weights["mask_non_trainable"])

    def apply_mask(self, weights, mask1, mask2):
        return weights * mask1 * mask2


def registry_and_build_op(layer):
    op_name = "masking_op"
    mask_op = MaskOperation(op_name)
    layer.registry_weight_operation("kernel", mask_op)
    layer.build((1,))


def get_model_for_test():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(5, 5, 1)),
            NNCFWrapper(tf.keras.layers.Conv2D(2, 3, activation="relu", name="layer1")),
            (tf.keras.layers.Dense(4, name="layer2")),
        ]
    )

    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            registry_and_build_op(layer)
    return model


def check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights):
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            actual_trainable_weights = [weight.name for weight in layer.trainable_weights]
            actual_non_trainable_weights = [weight.name for weight in layer.non_trainable_weights]
            assert actual_trainable_weights == ref_trainable_weights
            assert actual_non_trainable_weights == ref_non_trainable_weights


def test_wrapper_weights_freeze():
    model = get_model_for_test()

    # Initial state check
    ref_trainable_weights = ["kernel_mask_trainable:0", "layer1/kernel:0", "layer1/kernel:0", "layer1/bias:0"]
    ref_non_trainable_weights = ["kernel_mask_non_trainable:0"]
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    # Switching off whole layer from training
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.trainable = False
    ref_trainable_weights = []
    ref_non_trainable_weights = [
        "kernel_mask_non_trainable:0",
        "layer1/kernel:0",
        "layer1/bias:0",
        "kernel_mask_trainable:0",
        "layer1/kernel:0",
    ]
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    # Operation weights are enabled for training
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.set_ops_trainable(True)
    ref_trainable_weights = ["kernel_mask_trainable:0"]
    ref_non_trainable_weights = ["kernel_mask_non_trainable:0", "layer1/kernel:0", "layer1/bias:0", "layer1/kernel:0"]
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    # All weights are enabled for training
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.trainable = True
    ref_trainable_weights = ["kernel_mask_trainable:0", "layer1/kernel:0", "layer1/kernel:0", "layer1/bias:0"]
    ref_non_trainable_weights = ["kernel_mask_non_trainable:0"]
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)

    # Operation weights are disable for training
    for layer in model.layers:
        if isinstance(layer, NNCFWrapper):
            layer.set_ops_trainable(False)
    ref_trainable_weights = ["layer1/kernel:0", "layer1/kernel:0", "layer1/bias:0"]
    ref_non_trainable_weights = ["kernel_mask_non_trainable:0", "kernel_mask_trainable:0"]
    check_train_weights(model, ref_trainable_weights, ref_non_trainable_weights)
