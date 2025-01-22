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


def NASNet(
    input_shape=None,
    penultimate_filters=4032,
    num_blocks=6,
    stem_block_filters=96,
    skip_reduction=True,
    filter_multiplier=2,
    default_size=None,
):
    if backend.image_data_format() != "channels_last":
        raise AttributeError('The input data format "channels_last" is only supported.')

    if default_size is None:
        default_size = 331

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=True,
    )

    img_input = layers.Input(shape=input_shape)

    if penultimate_filters % (24 * (filter_multiplier**2)) != 0:
        raise ValueError(
            "For NASNet-A models, the `penultimate_filters` must be a multiple "
            "of 24 * (`filter_multiplier` ** 2). Current value: %d" % penultimate_filters
        )

    channel_dim = -1
    filters = penultimate_filters // 24

    x = layers.Conv2D(
        stem_block_filters,
        (3, 3),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        name="stem_conv1",
        kernel_initializer="he_normal",
    )(img_input)

    x = layers.BatchNormalization(axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="stem_bn1")(x)

    p = None
    x, p = _reduction_a_cell(x, p, filters // (filter_multiplier**2), block_id="stem_1")
    x, p = _reduction_a_cell(x, p, filters // filter_multiplier, block_id="stem_2")

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters, block_id="%d" % (i))

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier, block_id="reduce_%d" % (num_blocks))

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier, block_id="%d" % (num_blocks + i + 1))

    x, p0 = _reduction_a_cell(x, p, filters * filter_multiplier**2, block_id="reduce_%d" % (2 * num_blocks))

    p = p0 if not skip_reduction else p

    for i in range(num_blocks):
        x, p = _normal_a_cell(x, p, filters * filter_multiplier**2, block_id="%d" % (2 * num_blocks + i + 1))

    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    imagenet_utils.validate_activation("softmax", None)
    x = layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(img_input, x, name="NASNet")

    return model


def NASNetMobile(input_shape=None):
    return NASNet(
        input_shape,
        penultimate_filters=1056,
        num_blocks=4,
        stem_block_filters=32,
        skip_reduction=False,
        filter_multiplier=2,
        default_size=224,
    )


def NASNetLarge(input_shape=None):
    return NASNet(
        input_shape,
        penultimate_filters=4032,
        num_blocks=6,
        stem_block_filters=96,
        skip_reduction=True,
        filter_multiplier=2,
        default_size=331,
    )


def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), block_id=None):
    channel_dim = -1

    with backend.name_scope("separable_conv_block_%s" % block_id):
        x = layers.Activation("relu")(ip)
        if strides == (2, 2):
            x = layers.ZeroPadding2D(
                padding=imagenet_utils.correct_pad(x, kernel_size), name="separable_conv_1_pad_%s" % block_id
            )(x)
            conv_pad = "valid"
        else:
            conv_pad = "same"
        x = layers.SeparableConv2D(
            filters,
            kernel_size,
            strides=strides,
            name="separable_conv_1_%s" % block_id,
            padding=conv_pad,
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="separable_conv_1_bn_%s" % (block_id)
        )(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(
            filters,
            kernel_size,
            name="separable_conv_2_%s" % block_id,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization(
            axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="separable_conv_2_bn_%s" % (block_id)
        )(x)
    return x


def _adjust_block(p, ip, filters, block_id=None):
    channel_dim = -1
    img_dim = -2

    ip_shape = backend.int_shape(ip)

    if p is not None:
        p_shape = backend.int_shape(p)

    with backend.name_scope("adjust_block"):
        if p is None:
            p = ip

        elif p_shape[img_dim] != ip_shape[img_dim]:
            with backend.name_scope("adjust_reduction_block_%s" % block_id):
                p = layers.Activation("relu", name="adjust_relu_1_%s" % block_id)(p)
                p1 = layers.AveragePooling2D(
                    (1, 1), strides=(2, 2), padding="valid", name="adjust_avg_pool_1_%s" % block_id
                )(p)
                p1 = layers.Conv2D(
                    filters // 2,
                    (1, 1),
                    padding="same",
                    use_bias=False,
                    name="adjust_conv_1_%s" % block_id,
                    kernel_initializer="he_normal",
                )(p1)

                p2 = layers.ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = layers.Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = layers.AveragePooling2D(
                    (1, 1), strides=(2, 2), padding="valid", name="adjust_avg_pool_2_%s" % block_id
                )(p2)
                p2 = layers.Conv2D(
                    filters // 2,
                    (1, 1),
                    padding="same",
                    use_bias=False,
                    name="adjust_conv_2_%s" % block_id,
                    kernel_initializer="he_normal",
                )(p2)

                p = layers.concatenate([p1, p2], axis=channel_dim)
                p = layers.BatchNormalization(
                    axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="adjust_bn_%s" % block_id
                )(p)

        elif p_shape[channel_dim] != filters:
            with backend.name_scope("adjust_projection_block_%s" % block_id):
                p = layers.Activation("relu")(p)
                p = layers.Conv2D(
                    filters,
                    (1, 1),
                    strides=(1, 1),
                    padding="same",
                    name="adjust_conv_projection_%s" % block_id,
                    use_bias=False,
                    kernel_initializer="he_normal",
                )(p)
                p = layers.BatchNormalization(
                    axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="adjust_bn_%s" % block_id
                )(p)
    return p


def _normal_a_cell(ip, p, filters, block_id=None):
    channel_dim = -1

    with backend.name_scope("normal_A_block_%s" % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = layers.Activation("relu")(ip)
        h = layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            name="normal_conv_1_%s" % block_id,
            use_bias=False,
            kernel_initializer="he_normal",
        )(h)
        h = layers.BatchNormalization(
            axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="normal_bn_1_%s" % block_id
        )(h)

        with backend.name_scope("block_1"):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), block_id="normal_left1_%s" % block_id)
            x1_2 = _separable_conv_block(p, filters, block_id="normal_right1_%s" % block_id)
            x1 = layers.add([x1_1, x1_2], name="normal_add_1_%s" % block_id)

        with backend.name_scope("block_2"):
            x2_1 = _separable_conv_block(p, filters, (5, 5), block_id="normal_left2_%s" % block_id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), block_id="normal_right2_%s" % block_id)
            x2 = layers.add([x2_1, x2_2], name="normal_add_2_%s" % block_id)

        with backend.name_scope("block_3"):
            x3 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="normal_left3_%s" % (block_id))(h)
            x3 = layers.add([x3, p], name="normal_add_3_%s" % block_id)

        with backend.name_scope("block_4"):
            x4_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="normal_left4_%s" % (block_id))(
                p
            )
            x4_2 = layers.AveragePooling2D(
                (3, 3), strides=(1, 1), padding="same", name="normal_right4_%s" % (block_id)
            )(p)
            x4 = layers.add([x4_1, x4_2], name="normal_add_4_%s" % block_id)

        with backend.name_scope("block_5"):
            x5 = _separable_conv_block(h, filters, block_id="normal_left5_%s" % block_id)
            x5 = layers.add([x5, h], name="normal_add_5_%s" % block_id)

        x = layers.concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name="normal_concat_%s" % block_id)
    return x, ip


def _reduction_a_cell(ip, p, filters, block_id=None):
    channel_dim = -1

    with backend.name_scope("reduction_A_block_%s" % block_id):
        p = _adjust_block(p, ip, filters, block_id)

        h = layers.Activation("relu")(ip)
        h = layers.Conv2D(
            filters,
            (1, 1),
            strides=(1, 1),
            padding="same",
            name="reduction_conv_1_%s" % block_id,
            use_bias=False,
            kernel_initializer="he_normal",
        )(h)
        h = layers.BatchNormalization(
            axis=channel_dim, momentum=0.9997, epsilon=1e-3, name="reduction_bn_1_%s" % block_id
        )(h)
        h3 = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(h, 3), name="reduction_pad_1_%s" % block_id)(h)

        with backend.name_scope("block_1"):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), block_id="reduction_left1_%s" % block_id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id="reduction_right1_%s" % block_id)
            x1 = layers.add([x1_1, x1_2], name="reduction_add_1_%s" % block_id)

        with backend.name_scope("block_2"):
            x2_1 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="reduction_left2_%s" % block_id)(
                h3
            )
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), block_id="reduction_right2_%s" % block_id)
            x2 = layers.add([x2_1, x2_2], name="reduction_add_2_%s" % block_id)

        with backend.name_scope("block_3"):
            x3_1 = layers.AveragePooling2D(
                (3, 3), strides=(2, 2), padding="valid", name="reduction_left3_%s" % block_id
            )(h3)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), block_id="reduction_right3_%s" % block_id)
            x3 = layers.add([x3_1, x3_2], name="reduction_add3_%s" % block_id)

        with backend.name_scope("block_4"):
            x4 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding="same", name="reduction_left4_%s" % block_id)(
                x1
            )
            x4 = layers.add([x2, x4])

        with backend.name_scope("block_5"):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), block_id="reduction_left4_%s" % block_id)
            x5_2 = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="valid", name="reduction_right5_%s" % block_id)(
                h3
            )
            x5 = layers.add([x5_1, x5_2], name="reduction_add4_%s" % block_id)

        x = layers.concatenate([x2, x3, x4, x5], axis=channel_dim, name="reduction_concat_%s" % block_id)
        return x, ip
