"""
 Copyright (c) 2020 Intel Corporation
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
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import data_utils
import tensorflow_hub as hub


def mobilenet_v2_100_224(input_shape=None,
                         trainable=True,
                         batch_norm_momentum=0.997,
                         **_):
    handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/4"
    model = tf.keras.Sequential(
        [
            hub.KerasLayer(handle=handle, trainable=trainable,
                           arguments=dict(batch_norm_momentum=batch_norm_momentum)),
            tf.keras.layers.Activation('softmax')
        ]
    )

    input_shape = [None] + list(input_shape)
    model.build(input_shape)

    return model


def MobileNetV3(stack_fn, last_point_ch, input_shape=None, model_type='large', **_):
    if input_shape is None:
        input_shape = (None, None, 3)

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]
    if rows and cols and (rows < 32 or cols < 32):
        raise ValueError('Input size must be at least 32x32; got `input_shape=' +
                         str(input_shape) + '`')

    img_input = layers.Input(shape=input_shape)
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    kernel = 5
    activation = hard_swish
    se_ratio = 0.25

    x = img_input
    x = layers.Rescaling(scale=1. / 127.5, offset=-1.)(x)
    x = layers.Conv2D(
        16,
        kernel_size=3,
        strides=(2, 2),
        padding='same',
        use_bias=False,
        name='Conv')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv/BatchNorm')(x)
    x = activation(x)

    x = stack_fn(x, kernel, activation, se_ratio)

    last_conv_ch = _depth(backend.int_shape(x)[channel_axis] * 6)

    x = layers.Conv2D(
        last_conv_ch,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name='Conv_1')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, epsilon=1e-3,
        momentum=0.999, name='Conv_1/BatchNorm')(x)
    x = activation(x)

    x = layers.GlobalAveragePooling2D()(x)
    if channel_axis == 1:
        x = layers.Reshape((last_conv_ch, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, last_conv_ch))(x)

    x = layers.Conv2D(
        last_point_ch,
        kernel_size=1,
        padding='same',
        use_bias=True,
        name='Conv_2')(x)
    x = activation(x)

    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(1000, kernel_size=1, padding='same', name='Logits')(x)
    x = layers.Flatten()(x)
    x = layers.Activation(activation='softmax',
                          name='Predictions')(x)

    # Create model.
    model = models.Model(img_input, x, name='MobilenetV3{}'.format(model_type))

    BASE_WEIGHT_PATH = ('https://storage.googleapis.com/tensorflow/'
                        'keras-applications/mobilenet_v3/')
    WEIGHTS_HASHES = {
        'large': '59e551e166be033d707958cf9e29a6a7',
        'small': '8768d4c2e7dee89b9d02b2d03d65d862',
    }

    file_name = 'weights_mobilenet_v3_{}_224_1.0_float.h5'.format(model_type)
    file_hash = WEIGHTS_HASHES[model_type]

    weights_path = data_utils.get_file(
        file_name,
        BASE_WEIGHT_PATH + file_name,
        cache_subdir='models',
        file_hash=file_hash)
    model.load_weights(weights_path)

    return model


def MobileNetV3Small(input_shape=None, **kwargs):

    def stack_fn(x, kernel, activation, se_ratio):
        x = _inverted_res_block(x, 1, _depth(16), 3, 2, se_ratio, relu, 0)
        x = _inverted_res_block(x, 72. / 16, _depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 88. / 24, _depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 4, _depth(40), kernel, 2, se_ratio, activation, 3)
        x = _inverted_res_block(x, 6, _depth(40), kernel, 1, se_ratio, activation, 4)
        x = _inverted_res_block(x, 6, _depth(40), kernel, 1, se_ratio, activation, 5)
        x = _inverted_res_block(x, 3, _depth(48), kernel, 1, se_ratio, activation, 6)
        x = _inverted_res_block(x, 3, _depth(48), kernel, 1, se_ratio, activation, 7)
        x = _inverted_res_block(x, 6, _depth(96), kernel, 2, se_ratio, activation, 8)
        x = _inverted_res_block(x, 6, _depth(96), kernel, 1, se_ratio, activation, 9)
        x = _inverted_res_block(x, 6, _depth(96), kernel, 1, se_ratio, activation, 10)
        return x

    return MobileNetV3(stack_fn, 1024, input_shape, model_type='small', **kwargs)


def MobileNetV3Large(input_shape=None, **kwargs):

    def stack_fn(x, kernel, activation, se_ratio):
        x = _inverted_res_block(x, 1, _depth(16), 3, 1, None, relu, 0)
        x = _inverted_res_block(x, 4, _depth(24), 3, 2, None, relu, 1)
        x = _inverted_res_block(x, 3, _depth(24), 3, 1, None, relu, 2)
        x = _inverted_res_block(x, 3, _depth(40), kernel, 2, se_ratio, relu, 3)
        x = _inverted_res_block(x, 3, _depth(40), kernel, 1, se_ratio, relu, 4)
        x = _inverted_res_block(x, 3, _depth(40), kernel, 1, se_ratio, relu, 5)
        x = _inverted_res_block(x, 6, _depth(80), 3, 2, None, activation, 6)
        x = _inverted_res_block(x, 2.5, _depth(80), 3, 1, None, activation, 7)
        x = _inverted_res_block(x, 2.3, _depth(80), 3, 1, None, activation, 8)
        x = _inverted_res_block(x, 2.3, _depth(80), 3, 1, None, activation, 9)
        x = _inverted_res_block(x, 6, _depth(112), 3, 1, se_ratio, activation, 10)
        x = _inverted_res_block(x, 6, _depth(112), 3, 1, se_ratio, activation, 11)
        x = _inverted_res_block(x, 6, _depth(160), kernel, 2, se_ratio, activation, 12)
        x = _inverted_res_block(x, 6, _depth(160), kernel, 1, se_ratio, activation, 13)
        x = _inverted_res_block(x, 6, _depth(160), kernel, 1, se_ratio, activation, 14)
        return x

    return MobileNetV3(stack_fn, 1280, input_shape, model_type='large', **kwargs)


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([hard_sigmoid(x), x])


def _depth(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _se_block(inputs, filters, se_ratio, prefix):
    x = layers.GlobalAveragePooling2D(name=prefix + 'squeeze_excite/AvgPool')(
        inputs)
    if backend.image_data_format() == 'channels_first':
        x = layers.Reshape((filters, 1, 1))(x)
    else:
        x = layers.Reshape((1, 1, filters))(x)
    x = layers.Conv2D(
        _depth(filters * se_ratio),
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv')(
        x)
    x = layers.ReLU(name=prefix + 'squeeze_excite/Relu')(x)
    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        name=prefix + 'squeeze_excite/Conv_1')(
        x)
    x = hard_sigmoid(x)
    x = layers.Multiply(name=prefix + 'squeeze_excite/Mul')([inputs, x])
    return x


def _inverted_res_block(x, expansion, filters, kernel_size, stride, se_ratio,
                        activation, block_id):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    shortcut = x
    prefix = 'expanded_conv/'
    infilters = backend.int_shape(x)[channel_axis]
    if block_id:
        # Expand
        prefix = 'expanded_conv_{}/'.format(block_id)
        x = layers.Conv2D(
            _depth(infilters * expansion),
            kernel_size=1,
            padding='same',
            use_bias=False,
            name=prefix + 'expand')(
            x)
        x = layers.BatchNormalization(
            axis=channel_axis,
            epsilon=1e-3,
            momentum=0.999,
            name=prefix + 'expand/BatchNorm')(
            x)
        x = activation(x)

    if stride == 2:
        x = layers.ZeroPadding2D(
            padding=imagenet_utils.correct_pad(x, kernel_size),
            name=prefix + 'depthwise/pad')(
            x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=stride,
        padding='same' if stride == 1 else 'valid',
        use_bias=False,
        name=prefix + 'depthwise')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'depthwise/BatchNorm')(
        x)
    x = activation(x)

    if se_ratio:
        x = _se_block(x, _depth(infilters * expansion), se_ratio, prefix)

    x = layers.Conv2D(
        filters,
        kernel_size=1,
        padding='same',
        use_bias=False,
        name=prefix + 'project')(
        x)
    x = layers.BatchNormalization(
        axis=channel_axis,
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'project/BatchNorm')(
        x)

    if stride == 1 and infilters == filters:
        x = layers.Add(name=prefix + 'Add')([shortcut, x])
    return x
