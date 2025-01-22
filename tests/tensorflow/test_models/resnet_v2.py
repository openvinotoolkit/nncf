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


def ResNet50V2(input_shape=None):
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape, default_size=224, min_size=32, data_format=backend.image_data_format(), require_flatten=True
    )

    img_input = layers.Input(shape=input_shape)

    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    x = layers.Conv2D(64, 7, strides=2, use_bias=True, name="conv1_conv")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stack2(x, 64, 3, name="conv2")
    x = stack2(x, 128, 4, name="conv3")
    x = stack2(x, 256, 6, name="conv4")
    x = stack2(x, 512, 3, stride1=1, name="conv5")

    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name="post_bn")(x)
    x = layers.Activation("relu", name="post_relu")(x)

    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    imagenet_utils.validate_activation("softmax", None)
    x = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    # Create model.
    model = tf.keras.Model(img_input, x, name="resnet50v2")

    return model


def block2(x, filters, kernel_size=3, stride=1, conv_shortcut=False, name=None):
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1

    preact = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_preact_bn")(x)
    preact = layers.Activation("relu", name=name + "_preact_relu")(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(4 * filters, 1, strides=stride, name=name + "_0_conv")(preact)
    else:
        shortcut = layers.MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = layers.Conv2D(filters, 1, strides=1, use_bias=False, name=name + "_1_conv")(preact)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_1_bn")(x)
    x = layers.Activation("relu", name=name + "_1_relu")(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + "_2_pad")(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride, use_bias=False, name=name + "_2_conv")(x)
    x = layers.BatchNormalization(axis=bn_axis, epsilon=1.001e-5, name=name + "_2_bn")(x)
    x = layers.Activation("relu", name=name + "_2_relu")(x)

    x = layers.Conv2D(4 * filters, 1, name=name + "_3_conv")(x)
    x = layers.Add(name=name + "_out")([shortcut, x])
    return x


def stack2(x, filters, blocks, stride1=2, name=None):
    x = block2(x, filters, conv_shortcut=True, name=name + "_block1")
    for i in range(2, blocks):
        x = block2(x, filters, name=name + "_block" + str(i))
    x = block2(x, filters, stride=stride1, name=name + "_block" + str(blocks))
    return x
