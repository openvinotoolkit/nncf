"""
 Copyright (c) 2022 Intel Corporation
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

from typing import Any
from typing import Callable
from typing import Optional
from typing import Mapping

import tensorflow as tf

# pylint:disable=too-many-lines
# pylint:disable=too-many-statements
# pylint:disable=abstract-method
def make_divisible(value: float,
                   divisor: int,
                   min_value: Optional[float] = None,
                   round_down_protect: bool = True,
                   ) -> int:
    """This is to ensure that all layers have channels that are divisible by 8.

    Args:
        value: A `float` of original value.
        divisor: An `int` of the divisor that need to be checked upon.
        min_value: A `float` of  minimum value threshold.
        round_down_protect: A `bool` indicating whether round down more than 10%
            will be allowed.

    Returns:
        The adjusted value in `int` that is divisible against divisor.
    """
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)


class SqueezeExcitation(tf.keras.layers.Layer):
    """Creates a squeeze and excitation layer."""
    def __init__(self,
                 in_filters,
                 out_filters,
                 se_ratio,
                 divisible_by=1,
                 use_3d_input=False,
                 kernel_initializer='VarianceScaling',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation='relu',
                 gating_activation='sigmoid',
                 round_down_protect=True,
                 **kwargs):
        """Initializes a squeeze and excitation layer.

        Args:
          in_filters: An `int` number of filters of the input tensor.
          out_filters: An `int` number of filters of the output tensor.
          se_ratio: A `float` or None. If not None, se ratio for the squeeze and
            excitation layer.
          divisible_by: An `int` that ensures all inner dimensions are divisible by
            this number.
          use_3d_input: A `bool` of whether input is 2D or 3D image.
          kernel_initializer: A `str` of kernel_initializer for convolutional
            layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv2D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
            Default to None.
          activation: A `str` name of the activation function.
          gating_activation: A `str` name of the activation function for final
            gating function.
          round_down_protect: A `bool` of whether round down more than 10% will be
            allowed.
          **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(**kwargs)

        self._in_filters = in_filters
        self._out_filters = out_filters
        self._se_ratio = se_ratio
        self._divisible_by = divisible_by
        self._round_down_protect = round_down_protect
        self._use_3d_input = use_3d_input
        self._activation = activation
        self._gating_activation = gating_activation
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        if tf.keras.backend.image_data_format() == 'channels_last':
            if not use_3d_input:
                self._spatial_axis = [1, 2]
            else:
                self._spatial_axis = [1, 2, 3]
        else:
            if not use_3d_input:
                self._spatial_axis = [2, 3]
            else:
                self._spatial_axis = [2, 3, 4]
        self._activation_fn = get_activation(activation)
        self._gating_activation_fn = get_activation(gating_activation)

    def build(self, input_shape):
        num_reduced_filters = make_divisible(
            max(1, int(self._in_filters * self._se_ratio)),
            divisor=self._divisible_by,
            round_down_protect=self._round_down_protect)

        self._se_reduce = tf.keras.layers.Conv2D(
            filters=num_reduced_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)

        self._se_expand = tf.keras.layers.Conv2D(
            filters=self._out_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)

        super().build(input_shape)

    def get_config(self):
        config = {
            'in_filters': self._in_filters,
            'out_filters': self._out_filters,
            'se_ratio': self._se_ratio,
            'divisible_by': self._divisible_by,
            'use_3d_input': self._use_3d_input,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'activation': self._activation,
            'gating_activation': self._gating_activation,
            'round_down_protect': self._round_down_protect,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        x = tf.reduce_mean(inputs, self._spatial_axis, keepdims=True)
        x = self._activation_fn(self._se_reduce(x))
        x = self._gating_activation_fn(self._se_expand(x))
        return x * inputs


class StochasticDepth(tf.keras.layers.Layer):
    """Creates a stochastic depth layer."""

    def __init__(self, stochastic_depth_drop_rate, **kwargs):
        """Initializes a stochastic depth layer.

        Args:
          stochastic_depth_drop_rate: A `float` of drop rate.
          **kwargs: Additional keyword arguments to be passed.

        Returns:
          A output `tf.Tensor` of which should have the same shape as input.
        """
        super().__init__(**kwargs)
        self._drop_rate = stochastic_depth_drop_rate

    def get_config(self):
        config = {'drop_rate': self._drop_rate}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()
        if not training or self._drop_rate is None or self._drop_rate == 0:
            return inputs

        keep_prob = 1.0 - self._drop_rate
        batch_size = tf.shape(inputs)[0]
        random_tensor = keep_prob
        random_tensor += tf.random.uniform(
            [batch_size] + [1] * (inputs.shape.rank - 1), dtype=inputs.dtype)
        binary_tensor = tf.floor(random_tensor)
        output = tf.math.divide(inputs, keep_prob) * binary_tensor
        return output


def get_stochastic_depth_rate(init_rate, i, n):
    """Get drop connect rate for the ith block.

    Args:
      init_rate: A `float` of initial drop rate.
      i: An `int` of order of the current block.
      n: An `int` total number of blocks.

    Returns:
      Drop rate of the ith block.
    """
    if init_rate is not None:
        if init_rate < 0 or init_rate > 1:
            raise ValueError('Initial drop rate must be within 0 and 1.')
        rate = init_rate * float(i) / n
    else:
        rate = None
    return rate


def get_activation(identifier, use_keras_layer=False):
    """Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.

    It checks string first and if it is one of customized activation not in TF,
    the corresponding activation will be returned. For non-customized activation
    names and callable identifiers, always fallback to tf.keras.activations.get.

    Prefers using keras layers when use_keras_layer=True. Now it only supports
    'relu', 'linear', 'identity', 'swish'.

    Args:
      identifier: String name of the activation function or callable.
      use_keras_layer: If True, use keras layer if identifier is allow-listed.

    Returns:
      A Python function corresponding to the activation function or a keras
      activation layer when use_keras_layer=True.
    """
    if isinstance(identifier, str):
        identifier = str(identifier).lower()
        if use_keras_layer:
            keras_layer_allowlist = {
                "relu": "relu",
                "linear": "linear",
                "identity": "linear",
                "swish": "swish",
                "sigmoid": "sigmoid",
                "relu6": tf.nn.relu6,
                "hard_swish": None,
                "hard_sigmoid": None,
            }
            if identifier == 'relu':
                return tf.keras.layers.ReLU()
            if identifier in keras_layer_allowlist:
                return tf.keras.layers.Activation(keras_layer_allowlist[identifier])
        name_to_fn = {
            "gelu": None,
            "simple_swish": None,
            "hard_swish": None,
            "relu6": None,
            "hard_sigmoid": None,
            "identity": None,
        }
        assert identifier not in name_to_fn
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)


class ResidualBlock(tf.keras.layers.Layer):
    """A residual block."""

    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 se_ratio=None,
                 resnetd_shortcut=False,
                 stochastic_depth_drop_rate=None,
                 kernel_initializer='VarianceScaling',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation='relu',
                 use_explicit_padding: bool = False,
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 bn_trainable=True,
                 **kwargs):
        """Initializes a residual block with BN after convolutions.

        Args:
          filters: An `int` number of filters for the first two convolutions. Note
            that the third and final convolution will use 4 times as many filters.
          strides: An `int` block stride. If greater than 1, this block will
            ultimately downsample the input.
          use_projection: A `bool` for whether this block should use a projection
            shortcut (versus the default identity shortcut). This is usually `True`
            for the first block of a block group, which may change the number of
            filters and the resolution.
          se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
          resnetd_shortcut: A `bool` if True, apply the resnetd style modification
            to the shortcut connection. Not implemented in residual blocks.
          stochastic_depth_drop_rate: A `float` or None. if not None, drop rate for
            the stochastic depth layer.
          kernel_initializer: A `str` of kernel_initializer for convolutional
            layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv2D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
            Default to None.
          activation: A `str` name of the activation function.
          use_explicit_padding: Use 'VALID' padding for convolutions, but prepad
            inputs so that the output dimensions are the same as if 'SAME' padding
            were used.
          use_sync_bn: A `bool`. If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
          bn_trainable: A `bool` that indicates whether batch norm layers should be
            trainable. Default to True.
          **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(**kwargs)

        self._filters = filters
        self._strides = strides
        self._use_projection = use_projection
        self._se_ratio = se_ratio
        self._resnetd_shortcut = resnetd_shortcut
        self._use_explicit_padding = use_explicit_padding
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._kernel_initializer = kernel_initializer
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer

        if use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self._norm = tf.keras.layers.BatchNormalization
        if tf.keras.backend.image_data_format() == 'channels_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1
        self._activation_fn = get_activation(activation)
        self._bn_trainable = bn_trainable

    def build(self, input_shape):
        if self._use_projection:
            self._shortcut = tf.keras.layers.Conv2D(
                filters=self._filters,
                kernel_size=1,
                strides=self._strides,
                use_bias=False,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
            self._norm0 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable)

        conv1_padding = 'same'
        # explicit padding here is added for centernet
        if self._use_explicit_padding:
            self._pad = tf.keras.layers.ZeroPadding2D(padding=(1, 1))
            conv1_padding = 'valid'

        self._conv1 = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=3,
            strides=self._strides,
            padding=conv1_padding,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
        self._norm1 = self._norm(
            axis=self._bn_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            trainable=self._bn_trainable)

        self._conv2 = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=3,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
        self._norm2 = self._norm(
            axis=self._bn_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            trainable=self._bn_trainable)

        if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
            self._squeeze_excitation = SqueezeExcitation(
                in_filters=self._filters,
                out_filters=self._filters,
                se_ratio=self._se_ratio,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        else:
            self._squeeze_excitation = None

        if self._stochastic_depth_drop_rate:
            self._stochastic_depth = StochasticDepth(
                self._stochastic_depth_drop_rate)
        else:
            self._stochastic_depth = None

        super().build(input_shape)

    def get_config(self):
        config = {
            'filters': self._filters,
            'strides': self._strides,
            'use_projection': self._use_projection,
            'se_ratio': self._se_ratio,
            'resnetd_shortcut': self._resnetd_shortcut,
            'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'activation': self._activation,
            'use_explicit_padding': self._use_explicit_padding,
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon,
            'bn_trainable': self._bn_trainable
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        shortcut = inputs
        if self._use_projection:
            shortcut = self._shortcut(shortcut)
            shortcut = self._norm0(shortcut)

        if self._use_explicit_padding:
            inputs = self._pad(inputs)
        x = self._conv1(inputs)
        x = self._norm1(x)
        x = self._activation_fn(x)

        x = self._conv2(x)
        x = self._norm2(x)

        if self._squeeze_excitation:
            x = self._squeeze_excitation(x)

        if self._stochastic_depth:
            x = self._stochastic_depth(x, training=training)

        return self._activation_fn(x + shortcut)


class BottleneckBlock(tf.keras.layers.Layer):
    """A standard bottleneck block."""

    def __init__(self,
                 filters,
                 strides,
                 dilation_rate=1,
                 use_projection=False,
                 se_ratio=None,
                 resnetd_shortcut=False,
                 stochastic_depth_drop_rate=None,
                 kernel_initializer='VarianceScaling',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activation='relu',
                 use_sync_bn=False,
                 norm_momentum=0.99,
                 norm_epsilon=0.001,
                 bn_trainable=True,
                 **kwargs):
        """Initializes a standard bottleneck block with BN after convolutions.

        Args:
          filters: An `int` number of filters for the first two convolutions. Note
            that the third and final convolution will use 4 times as many filters.
          strides: An `int` block stride. If greater than 1, this block will
            ultimately downsample the input.
          dilation_rate: An `int` dilation_rate of convolutions. Default to 1.
          use_projection: A `bool` for whether this block should use a projection
            shortcut (versus the default identity shortcut). This is usually `True`
            for the first block of a block group, which may change the number of
            filters and the resolution.
          se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
          resnetd_shortcut: A `bool`. If True, apply the resnetd style modification
            to the shortcut connection.
          stochastic_depth_drop_rate: A `float` or None. If not None, drop rate for
            the stochastic depth layer.
          kernel_initializer: A `str` of kernel_initializer for convolutional
            layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv2D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2d.
            Default to None.
          activation: A `str` name of the activation function.
          use_sync_bn: A `bool`. If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A `float` added to variance to avoid dividing by zero.
          bn_trainable: A `bool` that indicates whether batch norm layers should be
            trainable. Default to True.
          **kwargs: Additional keyword arguments to be passed.
        """
        super().__init__(**kwargs)

        self._filters = filters
        self._strides = strides
        self._dilation_rate = dilation_rate
        self._use_projection = use_projection
        self._se_ratio = se_ratio
        self._resnetd_shortcut = resnetd_shortcut
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._stochastic_depth_drop_rate = stochastic_depth_drop_rate
        self._kernel_initializer = kernel_initializer
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        if use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self._norm = tf.keras.layers.BatchNormalization
        if tf.keras.backend.image_data_format() == 'channels_last':
            self._bn_axis = -1
        else:
            self._bn_axis = 1
        self._bn_trainable = bn_trainable

    def build(self, input_shape):
        if self._use_projection:
            if self._resnetd_shortcut:
                self._shortcut0 = tf.keras.layers.AveragePooling2D(
                    pool_size=2, strides=self._strides, padding='same')
                self._shortcut1 = tf.keras.layers.Conv2D(
                    filters=self._filters * 4,
                    kernel_size=1,
                    strides=1,
                    use_bias=False,
                    kernel_initializer=self._kernel_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer)
            else:
                self._shortcut = tf.keras.layers.Conv2D(
                    filters=self._filters * 4,
                    kernel_size=1,
                    strides=self._strides,
                    use_bias=False,
                    kernel_initializer=self._kernel_initializer,
                    kernel_regularizer=self._kernel_regularizer,
                    bias_regularizer=self._bias_regularizer)

            self._norm0 = self._norm(
                axis=self._bn_axis,
                momentum=self._norm_momentum,
                epsilon=self._norm_epsilon,
                trainable=self._bn_trainable)

        self._conv1 = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
        self._norm1 = self._norm(
            axis=self._bn_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            trainable=self._bn_trainable)
        self._activation1 = get_activation(
            self._activation, use_keras_layer=True)

        self._conv2 = tf.keras.layers.Conv2D(
            filters=self._filters,
            kernel_size=3,
            strides=self._strides,
            dilation_rate=self._dilation_rate,
            padding='same',
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
        self._norm2 = self._norm(
            axis=self._bn_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            trainable=self._bn_trainable)
        self._activation2 = get_activation(
            self._activation, use_keras_layer=True)

        self._conv3 = tf.keras.layers.Conv2D(
            filters=self._filters * 4,
            kernel_size=1,
            strides=1,
            use_bias=False,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer)
        self._norm3 = self._norm(
            axis=self._bn_axis,
            momentum=self._norm_momentum,
            epsilon=self._norm_epsilon,
            trainable=self._bn_trainable)
        self._activation3 = get_activation(
            self._activation, use_keras_layer=True)

        if self._se_ratio and self._se_ratio > 0 and self._se_ratio <= 1:
            self._squeeze_excitation = SqueezeExcitation(
                in_filters=self._filters * 4,
                out_filters=self._filters * 4,
                se_ratio=self._se_ratio,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)
        else:
            self._squeeze_excitation = None

        if self._stochastic_depth_drop_rate:
            self._stochastic_depth = StochasticDepth(
                self._stochastic_depth_drop_rate)
        else:
            self._stochastic_depth = None
        self._add = tf.keras.layers.Add()

        super().build(input_shape)

    def get_config(self):
        config = {
            'filters': self._filters,
            'strides': self._strides,
            'dilation_rate': self._dilation_rate,
            'use_projection': self._use_projection,
            'se_ratio': self._se_ratio,
            'resnetd_shortcut': self._resnetd_shortcut,
            'stochastic_depth_drop_rate': self._stochastic_depth_drop_rate,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'activation': self._activation,
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon,
            'bn_trainable': self._bn_trainable
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs, training=None):
        shortcut = inputs
        if self._use_projection:
            if self._resnetd_shortcut:
                shortcut = self._shortcut0(shortcut)
                shortcut = self._shortcut1(shortcut)
            else:
                shortcut = self._shortcut(shortcut)
            shortcut = self._norm0(shortcut)

        x = self._conv1(inputs)
        x = self._norm1(x)
        x = self._activation1(x)

        x = self._conv2(x)
        x = self._norm2(x)
        x = self._activation2(x)

        x = self._conv3(x)
        x = self._norm3(x)

        if self._squeeze_excitation:
            x = self._squeeze_excitation(x)

        if self._stochastic_depth:
            x = self._stochastic_depth(x, training=training)

        x = self._add([x, shortcut])
        return self._activation3(x)


# Specifications for different ResNet variants.
# Each entry specifies block configurations of the particular ResNet variant.
# Each element in the block configuration is in the following format:
# (block_fn, num_filters, block_repeats)
RESNET_SPECS = {
    10: [
        ('residual', 64, 1),
        ('residual', 128, 1),
        ('residual', 256, 1),
        ('residual', 512, 1),
    ],
    18: [
        ('residual', 64, 2),
        ('residual', 128, 2),
        ('residual', 256, 2),
        ('residual', 512, 2),
    ],
    34: [
        ('residual', 64, 3),
        ('residual', 128, 4),
        ('residual', 256, 6),
        ('residual', 512, 3),
    ],
    50: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 6),
        ('bottleneck', 512, 3),
    ],
    101: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 4),
        ('bottleneck', 256, 23),
        ('bottleneck', 512, 3),
    ],
    152: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 8),
        ('bottleneck', 256, 36),
        ('bottleneck', 512, 3),
    ],
    200: [
        ('bottleneck', 64, 3),
        ('bottleneck', 128, 24),
        ('bottleneck', 256, 36),
        ('bottleneck', 512, 3),
    ],
    270: [
        ('bottleneck', 64, 4),
        ('bottleneck', 128, 29),
        ('bottleneck', 256, 53),
        ('bottleneck', 512, 4),
    ],
    350: [
        ('bottleneck', 64, 4),
        ('bottleneck', 128, 36),
        ('bottleneck', 256, 72),
        ('bottleneck', 512, 4),
    ],
    420: [
        ('bottleneck', 64, 4),
        ('bottleneck', 128, 44),
        ('bottleneck', 256, 87),
        ('bottleneck', 512, 4),
    ],
}

#
@tf.keras.utils.register_keras_serializable(package='Vision')
class ResNet(tf.keras.Model):
    """Creates ResNet and ResNet-RS family models.

    This implements the Deep Residual Network from:
      Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
      Deep Residual Learning for Image Recognition.
      (https://arxiv.org/pdf/1512.03385) and
      Irwan Bello, William Fedus, Xianzhi Du, Ekin D. Cubuk, Aravind Srinivas,
      Tsung-Yi Lin, Jonathon Shlens, Barret Zoph.
      Revisiting ResNets: Improved Training and Scaling Strategies.
      (https://arxiv.org/abs/2103.07579).
    """

    def __init__(
        self,
        model_id: int,
        input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
            shape=[None, None, None, 3]),
        depth_multiplier: float = 1.0,
        stem_type: str = 'v0',
        resnetd_shortcut: bool = False,
        replace_stem_max_pool: bool = False,
        se_ratio: Optional[float] = None,
        init_stochastic_depth_rate: float = 0.0,
        scale_stem: bool = True,
        activation: str = 'relu',
        use_sync_bn: bool = False,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        kernel_initializer: str = 'VarianceScaling',
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bn_trainable: bool = True,
        **kwargs):
        """Initializes a ResNet model.

        Args:
          model_id: An `int` of the depth of ResNet backbone model.
          input_specs: A `tf.keras.layers.InputSpec` of the input tensor.
          depth_multiplier: A `float` of the depth multiplier to uniformaly scale up
            all layers in channel size. This argument is also referred to as
            `width_multiplier` in (https://arxiv.org/abs/2103.07579).
          stem_type: A `str` of stem type of ResNet. Default to `v0`. If set to
            `v1`, use ResNet-D type stem (https://arxiv.org/abs/1812.01187).
          resnetd_shortcut: A `bool` of whether to use ResNet-D shortcut in
            downsampling blocks.
          replace_stem_max_pool: A `bool` of whether to replace the max pool in stem
            with a stride-2 conv,
          se_ratio: A `float` or None. Ratio of the Squeeze-and-Excitation layer.
          init_stochastic_depth_rate: A `float` of initial stochastic depth rate.
          scale_stem: A `bool` of whether to scale stem layers.
          activation: A `str` name of the activation function.
          use_sync_bn: If True, use synchronized batch normalization.
          norm_momentum: A `float` of normalization momentum for the moving average.
          norm_epsilon: A small `float` added to variance to avoid dividing by zero.
          kernel_initializer: A str for kernel initializer of convolutional layers.
          kernel_regularizer: A `tf.keras.regularizers.Regularizer` object for
            Conv2D. Default to None.
          bias_regularizer: A `tf.keras.regularizers.Regularizer` object for Conv2D.
            Default to None.
          bn_trainable: A `bool` that indicates whether batch norm layers should be
            trainable. Default to True.
          **kwargs: Additional keyword arguments to be passed.
        """
        self._model_id = model_id
        self._input_specs = input_specs
        self._depth_multiplier = depth_multiplier
        self._stem_type = stem_type
        self._resnetd_shortcut = resnetd_shortcut
        self._replace_stem_max_pool = replace_stem_max_pool
        self._se_ratio = se_ratio
        self._init_stochastic_depth_rate = init_stochastic_depth_rate
        self._scale_stem = scale_stem
        self._use_sync_bn = use_sync_bn
        self._activation = activation
        self._norm_momentum = norm_momentum
        self._norm_epsilon = norm_epsilon
        if use_sync_bn:
            self._norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self._norm = tf.keras.layers.BatchNormalization
        self._kernel_initializer = kernel_initializer
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._bn_trainable = bn_trainable

        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = -1
        else:
            bn_axis = 1

        # Build ResNet.
        inputs = tf.keras.Input(shape=input_specs.shape[1:])

        stem_depth_multiplier = self._depth_multiplier if scale_stem else 1.0
        if stem_type == 'v0':
            x = tf.keras.layers.Conv2D(
                filters=int(64 * stem_depth_multiplier),
                kernel_size=7,
                strides=2,
                use_bias=False,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)(
                    inputs)
            x = self._norm(
                axis=bn_axis,
                momentum=norm_momentum,
                epsilon=norm_epsilon,
                trainable=bn_trainable)(
                    x)
            x = get_activation(activation, use_keras_layer=True)(x)
        elif stem_type == 'v1':
            x = tf.keras.layers.Conv2D(
                filters=int(32 * stem_depth_multiplier),
                kernel_size=3,
                strides=2,
                use_bias=False,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)(
                    inputs)
            x = self._norm(
                axis=bn_axis,
                momentum=norm_momentum,
                epsilon=norm_epsilon,
                trainable=bn_trainable)(
                    x)
            x = get_activation(activation, use_keras_layer=True)(x)
            x = tf.keras.layers.Conv2D(
                filters=int(32 * stem_depth_multiplier),
                kernel_size=3,
                strides=1,
                use_bias=False,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)(
                    x)
            x = self._norm(
                axis=bn_axis,
                momentum=norm_momentum,
                epsilon=norm_epsilon,
                trainable=bn_trainable)(
                    x)
            x = get_activation(activation, use_keras_layer=True)(x)
            x = tf.keras.layers.Conv2D(
                filters=int(64 * stem_depth_multiplier),
                kernel_size=3,
                strides=1,
                use_bias=False,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)(
                    x)
            x = self._norm(
                axis=bn_axis,
                momentum=norm_momentum,
                epsilon=norm_epsilon,
                trainable=bn_trainable)(
                    x)
            x = get_activation(activation, use_keras_layer=True)(x)
        else:
            raise ValueError('Stem type {} not supported.'.format(stem_type))

        if replace_stem_max_pool:
            x = tf.keras.layers.Conv2D(
                filters=int(64 * self._depth_multiplier),
                kernel_size=3,
                strides=2,
                use_bias=False,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer)(
                    x)
            x = self._norm(
                axis=bn_axis,
                momentum=norm_momentum,
                epsilon=norm_epsilon,
                trainable=bn_trainable)(
                    x)
            x = get_activation(activation, use_keras_layer=True)(x)
        else:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

        endpoints = {}
        for i, spec in enumerate(RESNET_SPECS[model_id]):
            if spec[0] == 'residual':
                block_fn = ResidualBlock
            elif spec[0] == 'bottleneck':
                block_fn = BottleneckBlock
            else:
                raise ValueError('Block fn `{}` is not supported.'.format(spec[0]))
            x = self._block_group(
                inputs=x,
                filters=int(spec[1] * self._depth_multiplier),
                strides=(1 if i == 0 else 2),
                block_fn=block_fn,
                block_repeats=spec[2],
                stochastic_depth_drop_rate=get_stochastic_depth_rate(
                    self._init_stochastic_depth_rate, i + 2, 5),
                name='block_group_l{}'.format(i + 2))
            endpoints[str(i + 2)] = x

        self._output_specs = {k: v.get_shape() for k, v in endpoints.items()}

        super().__init__(inputs=inputs, outputs=endpoints, **kwargs)

    def _block_group(self,
                     inputs: tf.Tensor,
                     filters: int,
                     strides: int,
                     block_fn: Callable[..., tf.keras.layers.Layer],
                     block_repeats: int = 1,
                     stochastic_depth_drop_rate: float = 0.0,
                     name: str = 'block_group'):
        """Creates one group of blocks for the ResNet model.

        Args:
          inputs: A `tf.Tensor` of size `[batch, channels, height, width]`.
          filters: An `int` number of filters for the first convolution of the
            layer.
          strides: An `int` stride to use for the first convolution of the layer.
            If greater than 1, this layer will downsample the input.
          block_fn: The type of block group. Either `ResidualBlock` or
            `BottleneckBlock`.
          block_repeats: An `int` number of blocks contained in the layer.
          stochastic_depth_drop_rate: A `float` of drop rate of the current block
            group.
          name: A `str` name for the block.

        Returns:
          The output `tf.Tensor` of the block layer.
        """
        x = block_fn(
            filters=filters,
            strides=strides,
            use_projection=True,
            stochastic_depth_drop_rate=stochastic_depth_drop_rate,
            se_ratio=self._se_ratio,
            resnetd_shortcut=self._resnetd_shortcut,
            kernel_initializer=self._kernel_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activation=self._activation,
            use_sync_bn=self._use_sync_bn,
            norm_momentum=self._norm_momentum,
            norm_epsilon=self._norm_epsilon,
            bn_trainable=self._bn_trainable)(
                inputs)

        for _ in range(1, block_repeats):
            x = block_fn(
                filters=filters,
                strides=1,
                use_projection=False,
                stochastic_depth_drop_rate=stochastic_depth_drop_rate,
                se_ratio=self._se_ratio,
                resnetd_shortcut=self._resnetd_shortcut,
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                activation=self._activation,
                use_sync_bn=self._use_sync_bn,
                norm_momentum=self._norm_momentum,
                norm_epsilon=self._norm_epsilon,
                bn_trainable=self._bn_trainable)(
                    x)

        return tf.keras.layers.Activation('linear', name=name)(x)

    def get_config(self):
        config_dict = {
            'model_id': self._model_id,
            'depth_multiplier': self._depth_multiplier,
            'stem_type': self._stem_type,
            'resnetd_shortcut': self._resnetd_shortcut,
            'replace_stem_max_pool': self._replace_stem_max_pool,
            'activation': self._activation,
            'se_ratio': self._se_ratio,
            'init_stochastic_depth_rate': self._init_stochastic_depth_rate,
            'scale_stem': self._scale_stem,
            'use_sync_bn': self._use_sync_bn,
            'norm_momentum': self._norm_momentum,
            'norm_epsilon': self._norm_epsilon,
            'kernel_initializer': self._kernel_initializer,
            'kernel_regularizer': self._kernel_regularizer,
            'bias_regularizer': self._bias_regularizer,
            'bn_trainable': self._bn_trainable
        }
        return config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @property
    def output_specs(self):
        """A dict of {level: TensorShape} pairs for the model output."""
        return self._output_specs


class ClassificationModel(tf.keras.Model):
    """A classification class builder."""

    def __init__(
        self,
        backbone: tf.keras.Model,
        num_classes: int,
        input_specs: tf.keras.layers.InputSpec = tf.keras.layers.InputSpec(
            shape=[None, None, None, 3]),
        dropout_rate: float = 0.0,
        kernel_initializer: str = 'random_uniform',
        kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Optional[tf.keras.regularizers.Regularizer] = None,
        add_head_batch_norm: bool = False,
        use_sync_bn: bool = False,
        norm_momentum: float = 0.99,
        norm_epsilon: float = 0.001,
        skip_logits_layer: bool = False,
        **kwargs):
        """Classification initialization function.

        Args:
          backbone: a backbone network.
          num_classes: `int` number of classes in classification task.
          input_specs: `tf.keras.layers.InputSpec` specs of the input tensor.
          dropout_rate: `float` rate for dropout regularization.
          kernel_initializer: kernel initializer for the dense layer.
          kernel_regularizer: tf.keras.regularizers.Regularizer object. Default to
                              None.
          bias_regularizer: tf.keras.regularizers.Regularizer object. Default to
                              None.
          add_head_batch_norm: `bool` whether to add a batch normalization layer
            before pool.
          use_sync_bn: `bool` if True, use synchronized batch normalization.
          norm_momentum: `float` normalization momentum for the moving average.
          norm_epsilon: `float` small float added to variance to avoid dividing by
            zero.
          skip_logits_layer: `bool`, whether to skip the prediction layer.
          **kwargs: keyword arguments to be passed.
        """
        if use_sync_bn:
            norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            norm = tf.keras.layers.BatchNormalization
        axis = -1 if tf.keras.backend.image_data_format() == 'channels_last' else 1

        inputs = tf.keras.Input(shape=input_specs.shape[1:], name=input_specs.name)
        endpoints = backbone(inputs)
        x = endpoints[max(endpoints.keys())]

        if add_head_batch_norm:
            x = norm(axis=axis, momentum=norm_momentum, epsilon=norm_epsilon)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if not skip_logits_layer:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            x = tf.keras.layers.Dense(
                num_classes,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer)(
                    x)

        super().__init__(
            inputs=inputs, outputs=x, **kwargs)
        self._config_dict = {
            'backbone': backbone,
            'num_classes': num_classes,
            'input_specs': input_specs,
            'dropout_rate': dropout_rate,
            'kernel_initializer': kernel_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'add_head_batch_norm': add_head_batch_norm,
            'use_sync_bn': use_sync_bn,
            'norm_momentum': norm_momentum,
            'norm_epsilon': norm_epsilon,
        }
        self._input_specs = input_specs
        self._kernel_regularizer = kernel_regularizer
        self._bias_regularizer = bias_regularizer
        self._backbone = backbone
        self._norm = norm

    @property
    def checkpoint_items(self) -> Mapping[str, tf.keras.Model]:
        """Returns a dictionary of items to be additionally checkpointed."""
        return dict(backbone=self.backbone)

    @property
    def backbone(self) -> tf.keras.Model:
        return self._backbone

    def get_config(self) -> Mapping[str, Any]:
        return self._config_dict

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


def resnet_50() -> tf.keras.Model:
    input_specs = tf.keras.layers.InputSpec(shape=(None, 224, 224, 3))
    l2_weight_decay = 0.0001
    l2_regularizer = (tf.keras.regularizers.l2(l2_weight_decay / 2.0) if l2_weight_decay else None)

    backbone = ResNet(
        model_id=50,
        input_specs=input_specs,
        depth_multiplier=1.0,
        stem_type='v0',
        resnetd_shortcut=False,
        replace_stem_max_pool=False,
        se_ratio=0.0,
        init_stochastic_depth_rate=0.0,
        scale_stem=True,
        activation='relu',
        use_sync_bn=False,
        norm_momentum=0.9,
        norm_epsilon=0.00001,
        kernel_regularizer=l2_regularizer,
        bn_trainable=True)

    model = ClassificationModel(
        backbone=backbone,
        num_classes=1001,
        input_specs=input_specs,
        dropout_rate=0.0,
        kernel_initializer='random_uniform',
        kernel_regularizer=l2_regularizer,
        add_head_batch_norm=False,
        use_sync_bn=False,
        norm_momentum=0.9,
        norm_epsilon=0.00001,
        skip_logits_layer=False
    )

    return model
