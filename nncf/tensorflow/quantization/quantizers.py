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

from functools import partial
from typing import Any
from typing import Dict
from typing import Optional

import tensorflow as tf

from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerSpec
from nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATIONS
from nncf.tensorflow.layers.data_layout import get_channel_axis
from nncf.tensorflow.layers.data_layout import get_channel_size
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.quantization.functions import asymmetric_quantize
from nncf.tensorflow.quantization.functions import symmetric_quantize


class TFQuantizerSpec(QuantizerSpec):
    def __init__(self, num_bits: int,
                 mode: QuantizationMode,
                 signedness_to_force: Optional[bool],
                 narrow_range: bool,
                 half_range: bool,
                 per_channel: bool):
        super().__init__(num_bits, mode, signedness_to_force, narrow_range, half_range)
        self.per_channel = per_channel

    @classmethod
    def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool) -> 'TFQuantizerSpec':
        return cls(qconfig.num_bits,
                   qconfig.mode,
                   qconfig.signedness_to_force,
                   narrow_range,
                   half_range,
                   qconfig.per_channel)

    def get_state(self) -> Dict[str, Any]:
        """
        Returns a dictionary with Python data structures (dict, list, tuple, str, int, float, True, False, None) that
        represents state of the object.

        :return: state of the object
        """
        return {
            'num_bits': self.num_bits,
            'mode': self.mode,
            'signedness_to_force': self.signedness_to_force,
            'narrow_range': self.narrow_range,
            'half_range': self.half_range,
            'per_channel': self.per_channel
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'TFQuantizerSpec':
        """
        Creates the object from its state.

        :param state: Output of `get_state()` method.
        """
        return cls(**state)


class Quantizer(NNCFOperation):
    """
    Base class for all NNCF quantization operations.
    """

    def __init__(self, name: str):
        """
        Initializes internal NNCF quantization operation state.

        :param name: Unique operation name in algorithm scope.
        """
        super().__init__(name)
        self.enabled = True
        self._eps = 1e-16
        self._pre_processing_fn = self._make_pre_processing_fn()
        self._post_processing_fn = self._make_post_processing_fn()

    @property
    def mode(self) -> str:
        """
        Returns mode of the quantization (symmetric or asymmetric).

        :return: The mode of the quantization.
        """
        raise NotImplementedError

    def call(self, inputs, weights, training):
        """
        The method applies quantization to the input tensor if the quantizer is enabled,
        otherwise, if the quantizer is disabled, the method returns the input tensor as is.

        :param inputs: Input tensor.
        :param weights: Quantizer's weights.
        :param training: True if operation called in training mode else False
        :return: Output tensor.
        """
        if not self.enabled:
            return inputs
        transformed = self._pre_processing_fn(inputs)
        quantized = self.quantize(transformed, weights, training)
        outputs = self._post_processing_fn(quantized)
        return outputs

    def quantize(self, inputs, weights, training):
        """
        Apply quantization to the input tensor.

        :param inputs: Input tensor.
        :param weights: Quantizer's weights.
        :param training: True if operation called in training mode else False
        :return: Quantized tensor.
        """
        raise NotImplementedError

    def apply_range_initialization(self, weights, min_values, max_values, min_range=0.1, eps=0.01):
        """
        Initialize quantizer parameters using minimum and maximum weight values.

        :param weights: Quantizer's weights.
        :param min_values: Minimum weight values.
        :param max_values: Maximum weight values.
        :param min_range: Minimum range.
        :param eps: Smoothing coefficient for ranges: min_range = maximum(min_range, eps * max_range).
        """
        raise NotImplementedError

    def setup_input_transformation(self, input_shape, channel_axes):
        """
        Setup input transformation that the per-channel quantization can be applied to input tensor.
        The TensorFlow fake_quant_with_min_max_vars_per_channel supports only inputs tensor one of
        the shapes: [d], [b, d] [b, h, w, d]. For this reason, Quantizer transforms any inputs tensor
        to one of the supported shapes, then quantizes and then transforms quantized tensor to
        the original inputs shape.

        :param input_shape: Shape of the input.
        :param channel_axes: Channel axes.
        """
        try:
            self._pre_processing_fn, self._post_processing_fn = \
                Quantizer._make_transformation_fns(input_shape, channel_axes)
        except NotImplementedError as e:
            raise NotImplementedError(f'Additional information: quantizer name {self.name}') from e

    @staticmethod
    def _make_transformation_fns(input_shape, channel_axes):
        fns_registry = []
        if isinstance(channel_axes, (tuple, list)):
            switch_counter = 0
            accumulate = False
            new_shape = []
            new_channel_axes = None
            for axis, val in enumerate(input_shape):
                if axis in channel_axes:
                    if accumulate:
                        new_shape[-1] *= val
                    else:
                        accumulate = True
                        new_channel_axes = len(new_shape)
                        new_shape.append(val)
                        switch_counter += 1
                else:
                    accumulate = False
                    new_shape.append(val)
            if switch_counter > 1:
                raise NotImplementedError(
                    'Quntizer could not transform input to apply per-channel quantization: '
                     f'input_shape {input_shape}, channel_axes {channel_axes}')
            forward_params = {'shape': new_shape}
            backward_params = {'shape': input_shape}
            fns_registry.append((tf.reshape, forward_params, backward_params))
            input_shape = new_shape
            channel_axes = new_channel_axes

        ndims = len(input_shape)
        if channel_axes % ndims != ndims - 1:
            perm = [i for i, _ in enumerate(input_shape)]
            perm[channel_axes], perm[-1] = perm[-1], perm[channel_axes]
            params = {'perm': perm}
            fns_registry.append((tf.transpose, params, params))
            new_shape = list(input_shape)
            new_shape[channel_axes], new_shape[-1] = new_shape[-1], new_shape[channel_axes]
            input_shape = new_shape

        if ndims not in [1, 2, 4]:
            size = 1
            for val in input_shape[:-1]:
                size *= val
            forward_params = {'shape': [size, input_shape[-1]]}
            backward_params = {'shape': input_shape}
            fns_registry.append((tf.reshape, forward_params, backward_params))

        def fuse_functions(fns_registry):
            if not fns_registry:
                return fns_registry

            fused_fns_registry = []
            fn1 = fns_registry[0]
            for fn2 in fns_registry[1:]:
                if fn1[0] == fn2[0] == tf.reshape:
                    fn1 = (tf.reshape, fn2[1], fn1[2])
                else:
                    fused_fns_registry.append(fn1)
                    fn1 = fn2
            fused_fns_registry.append(fn1)
            return fused_fns_registry

        fused_fns_registry = fuse_functions(fns_registry)
        pre_processing_fn = Quantizer._make_pre_processing_fn(fused_fns_registry)
        post_processing_fn = Quantizer._make_post_processing_fn(fused_fns_registry)

        return pre_processing_fn, post_processing_fn

    @staticmethod
    def _make_pre_processing_fn(fns_registry=None):
        fns_list = []
        if fns_registry is None:
            fns_registry = []
        for fn in fns_registry:
            fns_list.append(partial(fn[0], **fn[1]))

        def pre_processing_fn(inputs):
            result = inputs
            for func in fns_list:
                result = func(result)
            return result

        return pre_processing_fn

    @staticmethod
    def _make_post_processing_fn(fns_registry=None):
        fns_list = []
        if fns_registry is None:
            fns_registry = []
        for fn in reversed(fns_registry):
            fns_list.append(partial(fn[0], **fn[2]))

        def post_processing_fn(inputs):
            result = inputs
            for func in fns_list:
                result = func(result)
            return result

        return post_processing_fn

    @staticmethod
    def _min_adj(bits, low, range_len, narrow_range):
        quants_count = 2 ** bits - (2 if narrow_range else 1)
        return range_len / quants_count * tf.round(quants_count * low / range_len)

    def get_quantizer_config(self) -> QuantizerConfig:
        """
        Used to get a current quantizer state in terms of QuantizerConfig objects.

        :return: A QuantizerConfig struct that corresponds to current state of the quantizer.
        """
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError


@NNCF_CUSTOM_OBJECTS.register()
@NNCF_QUANTIZATION_OPERATIONS.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizer(Quantizer):
    def __init__(self, name: str, qspec: TFQuantizerSpec):
        super().__init__(name)
        self.num_bits = qspec.num_bits
        self.per_channel = qspec.per_channel
        self.narrow_range = qspec.narrow_range
        self.signedness_to_force = qspec.signedness_to_force
        self._half_range = qspec.half_range

    @property
    def half_range(self):
        return self._half_range

    @property
    def mode(self) -> str:
        return QuantizationMode.SYMMETRIC

    def signed(self, op_weights) -> bool:
        """
        Returns `True` for signed quantization, `False` for unsigned.

        :return: `True` for signed quantization, `False` for unsigned.
        """
        signed_var = op_weights['signed_var']
        return signed_var.numpy() < 0.0

    def build(self, input_shape, input_type, name, layer):
        channel_axes = None
        if self.per_channel:
            channel_axes = get_channel_axis(input_type, name, layer)
        return self._create_variables(layer, input_shape, channel_axes, name)

    def _create_variables(self,
                          layer,
                          input_shape,
                          channel_axes,
                          name: str = ''):
        shape = None
        if self.per_channel:
            self.setup_input_transformation(input_shape, channel_axes)
            shape = (get_channel_size(input_shape, channel_axes),)

        scale = layer.add_weight(
            name + '_scale',
            shape=shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        signed = layer.add_weight(
            name + '_signed',
            initializer=tf.keras.initializers.Constant(
                -1.0 if self.signedness_to_force in (True, None) else 0.0),
            trainable=False)
        return {
            'scale_var': scale,
            'signed_var': signed
        }

    def apply_overflow_fix(self, weights):
        if self.num_bits != 8 or not self._half_range:
            raise RuntimeError('Attempt to apply overflow issue fix '
                               'to quantizer which is not configured for that.')

        # Multiplier to expand scale from 7 bit to 8 bit
        multiplier = 127 / 63 if self.narrow_range else 255 / 127
        weights['scale_var'].assign(multiplier * weights['scale_var'])
        self._eps *= multiplier
        self._half_range = False

    def quantize(self, inputs, weights, _):
        def _half_range_quantize():
            return symmetric_quantize(
                inputs,
                weights['scale_var'],
                weights['signed_var'],
                num_bits=self.num_bits - 1,
                per_channel=self.per_channel,
                narrow_range=self.narrow_range,
                eps=self._eps
            )

        def _default_quantize():
            return symmetric_quantize(
                inputs,
                weights['scale_var'],
                weights['signed_var'],
                num_bits=self.num_bits,
                per_channel=self.per_channel,
                narrow_range=self.narrow_range,
                eps=self._eps
            )

        if self._half_range:
            return _half_range_quantize()

        return _default_quantize()

    def apply_range_initialization(self, weights, min_values, max_values, min_range=0.1, eps=0.01):
        if self.signedness_to_force is None:
            sign = tf.reduce_any(tf.less(min_values, 0))
            weights['signed_var'].assign(-1.0 if sign else 0.0)
        ranges = tf.maximum(tf.abs(max_values), tf.abs(min_values))
        max_range = tf.reduce_max(ranges)
        lower_threshold = tf.maximum(eps * max_range, min_range)
        scale = tf.maximum(ranges, lower_threshold)
        weights['scale_var'].assign(scale)

    def get_quantizer_config(self) -> QuantizerConfig:
        return QuantizerConfig(
            num_bits=self.num_bits,
            mode=QuantizationMode.SYMMETRIC,
            signedness_to_force=self.signedness_to_force,
            per_channel=self.per_channel
        )

    def get_config(self):
        qspec_dict = {
            'num_bits':  self.num_bits,
            'mode': QuantizationMode.SYMMETRIC,
            'signedness_to_force': self.signedness_to_force,
            'narrow_range': self.narrow_range,
            'half_range': self._half_range,
            'per_channel': self.per_channel,
        }
        config = {
            'quantizer_spec': qspec_dict,
            'name': self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        qspec_dict = config['quantizer_spec']
        qspec = TFQuantizerSpec(num_bits=qspec_dict['num_bits'],
                                mode=QuantizationMode.SYMMETRIC,
                                signedness_to_force=qspec_dict['signedness_to_force'],
                                narrow_range=qspec_dict['narrow_range'],
                                half_range=qspec_dict['half_range'],
                                per_channel=qspec_dict['per_channel'])
        name = config['name']
        return cls(name, qspec)



@NNCF_CUSTOM_OBJECTS.register()
@NNCF_QUANTIZATION_OPERATIONS.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(Quantizer):
    def __init__(self, name: str, qspec: TFQuantizerSpec):
        super().__init__(name)
        self.num_bits = qspec.num_bits
        self.narrow_range = qspec.narrow_range
        self.per_channel = qspec.per_channel
        self._half_range = qspec.half_range

    @property
    def half_range(self):
        return self._half_range

    @property
    def mode(self) -> str:
        return QuantizationMode.ASYMMETRIC

    def build(self, input_shape, input_type, name, layer):
        channel_axes = None
        if self.per_channel:
            channel_axes = get_channel_axis(input_type, name, layer)
        return self._create_variables(layer, input_shape, channel_axes, name)

    def _create_variables(self,
                          layer,
                          input_shape,
                          channel_axes,
                          name: str = ''):
        shape = None
        if self.per_channel:
            self.setup_input_transformation(input_shape, channel_axes)
            shape = (get_channel_size(input_shape, channel_axes),)

        input_low = layer.add_weight(
            name + '_input_low',
            shape=shape,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True)
        input_range = layer.add_weight(
            name + '_input_range',
            shape=shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True)
        return {
            'input_low_var': input_low,
            'input_range_var': input_range
        }

    def apply_overflow_fix(self, weights):
        if self.num_bits != 8 or not self._half_range:
            raise RuntimeError('Attempt to apply overflow issue fix '
                               'to quantizer which is not configured for that.')

        # Low value shift to expand quantize range from 7 bit to 8 bit properly
        weights['input_low_var'].assign(weights['input_low_var'] + self._min_adj(
                                        7, weights['input_low_var'],
                                        weights['input_range_var'] + self._eps,
                                        self.narrow_range))
        # Multiplier to expand scale from 7 bit to 8 bit
        multiplier = 127 / 63 if self.narrow_range else 255 / 127
        weights['input_range_var'].assign(multiplier * weights['input_range_var'])
        self._eps *= multiplier
        self._half_range = False

    def quantize(self, inputs, weights, _):
        def _half_range_quantize():
            return asymmetric_quantize(
                inputs,
                weights['input_low_var'],
                weights['input_range_var'],
                num_bits=self.num_bits - 1,
                per_channel=self.per_channel,
                narrow_range=self.narrow_range,
                eps=self._eps
            )

        def _default_quantize():
            return asymmetric_quantize(
                inputs,
                weights['input_low_var'],
                weights['input_range_var'],
                num_bits=self.num_bits,
                per_channel=self.per_channel,
                narrow_range=self.narrow_range,
                eps=self._eps
            )

        if self._half_range:
            return _half_range_quantize()

        return _default_quantize()

    def apply_range_initialization(self, weights, min_values, max_values, min_range=0.1, eps=0.01):
        ranges = max_values - min_values
        max_range = tf.reduce_max(ranges)
        lower_threshold = tf.maximum(eps * max_range, min_range)
        correction = (tf.maximum(ranges, lower_threshold) - ranges) * 0.5
        input_low = min_values - correction
        input_range = ranges + 2 * correction
        weights['input_low_var'].assign(input_low)
        weights['input_range_var'].assign(input_range)

    def get_quantizer_config(self) -> QuantizerConfig:
        return QuantizerConfig(
            num_bits=self.num_bits,
            mode=QuantizationMode.ASYMMETRIC,
            signedness_to_force=None,
            per_channel=self.per_channel
        )

    def get_config(self):
        qspec_dict = {
            'num_bits': self.num_bits,
            'mode': QuantizationMode.ASYMMETRIC,
            'signedness_to_force': None,
            'narrow_range': self.narrow_range,
            'half_range': self._half_range,
            'per_channel': self.per_channel,
        }
        config = {
            'quantizer_spec': qspec_dict,
            'name': self.name,
        }
        return config

    @classmethod
    def from_config(cls, config):
        qspec_dict = config['quantizer_spec']
        qspec = TFQuantizerSpec(num_bits=qspec_dict['num_bits'],
                                mode=QuantizationMode.ASYMMETRIC,
                                signedness_to_force=None,
                                narrow_range=qspec_dict['narrow_range'],
                                half_range=qspec_dict['half_range'],
                                per_channel=qspec_dict['per_channel'])
        name = config['name']
        return cls(name, qspec)
