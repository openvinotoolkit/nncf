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

from nncf.tensorflow.tf_internals import backend
from nncf.tensorflow.tf_internals import imagenet_utils
from nncf.tensorflow.tf_internals import layers

NUM_CLASSES = 1000


def dense_block(x, blocks, name):
    for i in range(blocks):
        x = conv_block(x, 32, name=name + "_block" + str(i + 1))
    return x


def transition_block(x, reduction, name):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_bn")(x)
    x = layers.Activation("relu", name=name + "_relu")(x)
    x = layers.Conv2D(int(backend.int_shape(x)[bn_axis] * reduction), 1, use_bias=False, name=name + "_conv")(x)
    x = layers.AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x


def conv_block(x, growth_rate, name):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_0_bn")(x)
    x1 = layers.Activation("relu", name=name + "_0_relu")(x1)
    x1 = layers.Conv2D(4 * growth_rate, 1, use_bias=False, name=name + "_1_conv")(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x1)
    x1 = layers.Activation("relu", name=name + "_1_relu")(x1)
    x1 = layers.Conv2D(growth_rate, 3, padding="same", use_bias=False, name=name + "_2_conv")(x1)
    x = layers.Concatenate(axis=bn_axis, name=name + "_concat")([x, x1])
    return x


def DenseNet121(input_shape=None):
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape, default_size=224, min_size=32, data_format=backend.image_data_format(), require_flatten=True
    )

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="conv1/conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="conv1/bn")(x)
    x = layers.Activation("relu", name="conv1/relu")(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)

    x = dense_block(x, 6, name="conv2")
    x = transition_block(x, 0.5, name="pool2")
    x = dense_block(x, 12, name="conv3")
    x = transition_block(x, 0.5, name="pool3")
    x = dense_block(x, 24, name="conv4")
    x = transition_block(x, 0.5, name="pool4")
    x = dense_block(x, 16, name="conv5")

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="bn")(x)
    x = layers.Activation("relu", name="relu")(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)

    imagenet_utils.validate_activation("softmax", None)
    x = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    # Create model.
    model = tf.keras.Model(img_input, x, name="densenet121")

    return model
