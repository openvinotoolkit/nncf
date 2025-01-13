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
from tensorflow.keras import layers

from nncf import NNCFConfig


def get_basic_pruning_config(model_size=8):
    config = NNCFConfig()
    config.update(
        {
            "model": "basic",
            "input_info": {
                "sample_size": [1, model_size, model_size, 1],
            },
            "compression": {
                "algorithm": "filter_pruning",
                "pruning_init": 0.5,
                "params": {
                    "prune_first_conv": True,
                },
            },
        }
    )
    return config


def get_concat_test_model(input_shape):
    #             (input)
    #                |
    #             (conv1)
    #        /       |       \
    #    (conv2)  (conv3)  (conv4)
    #       |        |       |
    #       |    (gr_conv)   |
    #         \    /         |
    #        (concat)    (bn_conv4)
    #             \       /
    #              (concat)
    #                 |
    #            (bn_concat)
    #                 |
    #              (conv5)

    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    conv1 = layers.Conv2D(16, 1, name="conv1")
    conv2 = layers.Conv2D(16, 1, name="conv2")
    conv3 = layers.Conv2D(16, 1, name="conv3")
    group_conv = layers.Conv2D(16, 1, groups=8, name="group_conv")
    conv4 = layers.Conv2D(32, 1, name="conv4")
    bn_conv4 = layers.BatchNormalization(name="bn_conv4")
    bn_concat = layers.BatchNormalization(name="bn_concat")
    conv5 = layers.Conv2D(64, 1, name="conv5")

    x = conv1(inputs)
    x1 = tf.concat([conv2(x), group_conv(conv3(x))], -1, name="tf_concat_1")
    x = conv4(x)
    x = bn_conv4(x)
    x = tf.concat([x, x1], -1, name="tf_concat_2")
    x = bn_concat(x)
    outputs = conv5(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def init_conv_weights(weights):
    new_weights = weights.numpy()
    for i in range(weights.shape[-1]):
        new_weights[..., i] += i
    weights.assign(new_weights)


def get_test_model_shared_convs(input_shape):
    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    conv1 = layers.Conv2D(512, 1, name="conv1", kernel_initializer="Ones", bias_initializer="Ones")
    conv2 = layers.Conv2D(1024, 3, name="conv2", kernel_initializer="Ones", bias_initializer="Ones")
    conv3 = layers.Conv2D(1024, 1, name="conv3", kernel_initializer="Ones", bias_initializer="Ones")
    maxpool = layers.MaxPool2D()

    in1 = conv1(inputs)
    in2 = maxpool(in1)
    out1 = conv2(in1)
    out2 = conv2(in2)
    x = conv3(out1)
    y = conv3(out2)

    init_conv_weights(conv1.kernel)
    init_conv_weights(conv2.kernel)
    init_conv_weights(conv3.kernel)
    return tf.keras.Model(inputs=inputs, outputs=[x, y])


def get_model_grouped_convs(input_shape):
    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    conv1 = layers.Conv2D(512, 1, name="conv1", kernel_initializer="Ones", bias_initializer="Ones")
    conv2 = layers.Conv2D(128, 3, groups=4, name="conv2", kernel_initializer="Ones", bias_initializer="Ones")
    conv3 = layers.Conv2D(128, 3, groups=128, name="conv3", kernel_initializer="Ones", bias_initializer="Ones")
    conv4 = layers.DepthwiseConv2D(3, name="conv4", kernel_initializer="Ones", bias_initializer="Ones")
    flatten = layers.Flatten()
    linear = layers.Dense(128)

    x = conv1(inputs)
    x = conv2(x)
    x = conv3(x)
    x = conv4(x)
    x = linear(flatten(x))

    return tf.keras.Model(inputs=inputs, outputs=[x])


def get_model_depthwise_conv(input_shape):
    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    conv1 = layers.Conv2D(8, 1, name="conv1", kernel_initializer="Ones", bias_initializer="Ones")
    conv2 = layers.Conv2D(128, 3, name="conv2", kernel_initializer="Ones", bias_initializer="Ones")
    conv_depthwise = layers.DepthwiseConv2D(3, name="conv4", kernel_initializer="Ones", bias_initializer="Ones")
    conv3 = layers.Conv2D(8, 3, name="conv3", kernel_initializer="Ones", bias_initializer="Ones")
    flatten = layers.Flatten()
    linear = layers.Dense(128)

    x = conv1(inputs)
    x = conv2(x)
    x = conv_depthwise(x)
    x = conv3(x)
    x = linear(flatten(x))

    return tf.keras.Model(inputs=inputs, outputs=[x])


def get_split_test_model(input_shape):
    #          (input)
    #             |
    #          (conv1)
    #             |
    #          (chunk)
    #        /       \
    #    (conv2)  (conv3)
    #         \    /
    #        (concat)
    #           |
    #        (conv4)

    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    conv1 = layers.Conv2D(16, 1, name="conv1")
    conv2 = layers.Conv2D(32, 1, name="conv2")
    conv3 = layers.Conv2D(32, 1, name="conv3")
    conv4 = layers.Conv2D(64, 1, name="conv5")

    x = conv1(inputs)
    y1, x = tf.split(x, 2, -1, name="tf_split")
    y1 = conv2(y1)
    x = conv3(x)
    x = tf.concat([y1, x], -1, name="tf_concat")
    outputs = conv4(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_broadcasted_linear_model(input_shape):
    input_shape = [1, 8, 8, 1]
    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    first_conv = layers.Conv2D(32, 1, name="first_conv", kernel_initializer="Ones", bias_initializer="Ones")
    conv1 = layers.Conv2D(16, 1, name="conv1", kernel_initializer="Ones", bias_initializer="Ones")
    linear1 = layers.Dense(16, name="linear1", kernel_initializer="Ones", bias_initializer="Ones")
    last_conv = layers.Conv2D(16, 1, name="last_linear", kernel_initializer="Ones", bias_initializer="Ones")

    x = first_conv(inputs)
    y = conv1(x)
    z = linear1(tf.reshape(x, (-1, tf.reduce_prod(x.shape[1:]))))
    x = y + tf.reshape(z, (-1, 1, 1, 16))
    out = last_conv(x)
    return tf.keras.Model(inputs=inputs, outputs=[out])


def get_batched_linear_model(input_shape):
    input_shape = [1, 8, 8, 1]
    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    first_conv = layers.Conv2D(32, 1, name="first_conv", kernel_initializer="Ones", bias_initializer="Ones")
    linear1 = layers.Dense(16, name="linear1", kernel_initializer="Ones", bias_initializer="Ones")
    last_linear = layers.Dense(1, name="last_linear", kernel_initializer="Ones", bias_initializer="Ones")

    x = first_conv(inputs)
    x = linear1(x)
    out = last_linear(tf.reshape(x, (-1, tf.reduce_prod(x.shape[1:]))))
    return tf.keras.Model(inputs=inputs, outputs=[out])


def get_diff_cluster_channels_model(input_shape):
    input_shape = [1, 8, 8, 1]
    inputs = tf.keras.Input(shape=input_shape[1:], name="input")
    first_conv = layers.Conv2D(32, 1, name="first_conv", kernel_initializer="Ones", bias_initializer="Ones")
    conv1 = layers.Conv2D(16, 1, name="conv1", kernel_initializer="Ones", bias_initializer="Ones")
    linear1 = layers.Dense(2048, name="linear1", kernel_initializer="Ones", bias_initializer="Ones")
    last_conv = layers.Dense(16, name="last_linear", kernel_initializer="Ones", bias_initializer="Ones")

    x = first_conv(inputs)
    y = tf.reshape(conv1(x), (-1, tf.reduce_prod(x.shape[1:])))
    z = linear1(tf.reshape(x, (-1, tf.reduce_prod(x.shape[1:]))))
    out = last_conv(y + z)
    return tf.keras.Model(inputs=inputs, outputs=[out])
