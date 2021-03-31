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

from functools import partial
from typing import Optional
from copy import deepcopy

import tensorflow as tf

from beta.nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from beta.nncf.tensorflow.layers.custom_objects import NNCF_QUANTIZATION_OPERATONS
from beta.nncf.tensorflow.layers.data_layout import get_channel_axis
from beta.nncf.tensorflow.layers.data_layout import get_channel_size
from beta.nncf.tensorflow.layers.operation import NNCFOperation
from beta.nncf.tensorflow.quantization.functions import asymmetric_quantize
from beta.nncf.tensorflow.quantization.functions import symmetric_quantize
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerSpec


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
    def from_config(cls, qconfig: QuantizerConfig, narrow_range: bool, half_range: bool):
        return cls(qconfig.num_bits,
                   qconfig.mode,
                   qconfig.signedness_to_force,
                   narrow_range,
                   half_range,
                   qconfig.per_channel)


class Quantizer(NNCFOperation):
    """
    Base class for all NNCF quantization operations
    """
    def __init__(self):
        """
        Initializes internal NNCF quantization operation state
        """
        super().__init__()
        self.enabled = True
        self._eps = 1e-16
        self._pre_processing_fn = self._make_pre_processing_fn()
        self._post_processing_fn = self._make_post_processing_fn()

    def call(self, inputs, weights, _):
        """
        The method applies quantization to the input tensor if the quantizer is enabled,
        otherwise, if the quantizer is disabled, the method returns the input tensor as is

        :param inputs: input tensor
        :param weights: quantizer's weights
        :return: output tensor
        """
        if not self.enabled:
            return inputs
        transformed = self._pre_processing_fn(inputs)
        quantized = self.quantize(transformed, weights)
        outputs = self._post_processing_fn(quantized)
        return outputs

    def quantize(self, inputs, weights):
        """
        Apply quantization to the input tensor.

        :param inputs: input tensor
        :param weights: quantizer's weights
        :return: quantized tensor
        """
        raise NotImplementedError

    def apply_minmax_initialization(self, weights, min_values, max_values, min_range=0.1, eps=0.01):
        """
        Initialize quantizer parameters using minimum and maximum weight values

        :param weights: quantizer's weights
        :param min_values: minimum weight values
        :param max_values: maximum weight values
        :param min_range: minimum range
        :param eps: smoothing coefficient for ranges: min_range = maximum(min_range, eps * max_range)
        """
        raise NotImplementedError

    def setup_input_transformation(self, input_shape, input_type, input_name, layer):
        """
        Setup input transformation that the per-channel quantization can be applied to input tensor.
        The TensorFlow fake_quant_with_min_max_vars_per_channel supports only inputs tensor one of
        the shapes: [d], [b, d] [b, h, w, d]. For this reason, Quantizer transforms any inputs tensor
        to one of the supported shapes, then quantizes and then transforms quantized tensor to
        the original inputs shape

        :param input_shape: shape of the input
        :param input_type: type of the input identifies that inputs are layer weights
                           or inputs of the layer
        :param input_name: input name
        :param layer: layer, where the Quantizer is registered
        """
        self._pre_processing_fn, self._post_processing_fn = \
            self._make_transformation_fns(input_shape, input_type, input_name, layer)

    def _make_transformation_fns(self, input_shape, input_type, input_name, layer):
        channel_axes = get_channel_axis(input_type, input_name, layer)

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
                    'input shape {}, input type {}, input name {}, channel_axes {} '
                    'from layer {}'.format(
                        input_shape, input_type, input_name, channel_axes, layer.name))
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
        return self._make_pre_processing_fn(fused_fns_registry), self._make_post_processing_fn(fused_fns_registry)

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

    def get_quantizer_config(self) -> QuantizerConfig:
        """
        Used to get a current quantizer state in terms of QuantizerConfig objects.
        :return: A QuantizerConfig struct that corresponds to current state of the quantizer.
        """
        raise NotImplementedError

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        return cls()


@NNCF_CUSTOM_OBJECTS.register()
@NNCF_QUANTIZATION_OPERATONS.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizer(Quantizer):
    def __init__(self, qspec: TFQuantizerSpec):
        super().__init__()
        self.num_bits = qspec.num_bits
        self.per_channel = qspec.per_channel
        self.narrow_range = qspec.narrow_range
        self.signedness_to_force = qspec.signedness_to_force
        self._half_range = qspec.half_range

    def build(self, input_shape, input_type, name, layer):
        shape = None
        if self.per_channel:
            self.setup_input_transformation(input_shape, input_type, name, layer)
            shape = (get_channel_size(input_shape, input_type, name, layer),)

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

    def quantize(self, inputs, weights):
        return symmetric_quantize(
            inputs,
            weights['scale_var'],
            weights['signed_var'],
            num_bits=self.num_bits,
            per_channel=self.per_channel,
            narrow_range=self.narrow_range,
            eps=self._eps
        )

    def apply_minmax_initialization(self, weights, min_values, max_values, min_range=0.1, eps=0.01):
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
        qspec = TFQuantizerSpec(num_bits=self.num_bits,
                                mode=QuantizationMode.SYMMETRIC,
                                signedness_to_force=self.signedness_to_force,
                                narrow_range=self.narrow_range,
                                half_range=self._half_range ,
                                per_channel=self.per_channel)
        return deepcopy(qspec.__dict__)

    @classmethod
    def from_config(cls, config):
        qspec = TFQuantizerSpec(num_bits=config["num_bits"],
                                mode=QuantizationMode.SYMMETRIC,
                                signedness_to_force=config["signedness_to_force"],
                                narrow_range=config["narrow_range"],
                                half_range=config['half_range'],
                                per_channel=config["per_channel"])
        return cls(qspec)



@NNCF_CUSTOM_OBJECTS.register()
@NNCF_QUANTIZATION_OPERATONS.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(Quantizer):
    def __init__(self, qspec: TFQuantizerSpec):
        super().__init__()
        self.num_bits = qspec.num_bits
        self.narrow_range = qspec.narrow_range
        self.per_channel = qspec.per_channel
        self._half_range = qspec.half_range

    def build(self, input_shape, input_type, name, layer):
        shape = None
        if self.per_channel:
            self.setup_input_transformation(input_shape, input_type, name, layer)
            shape = (get_channel_size(input_shape, input_type, name, layer),)

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

    def quantize(self, inputs, weights):
        return asymmetric_quantize(
            inputs,
            weights['input_low_var'],
            weights['input_range_var'],
            num_bits=self.num_bits,
            per_channel=self.per_channel,
            narrow_range=self.narrow_range,
            eps=self._eps
        )

    def apply_minmax_initialization(self, weights, min_values, max_values, min_range=0.1, eps=0.01):
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
        qspec = TFQuantizerSpec(self.num_bits,
                                QuantizationMode.ASYMMETRIC,
                                signedness_to_force=None,
                                narrow_range=self.narrow_range,
                                half_range=self._half_range ,
                                per_channel=self.per_channel)
        return deepcopy(qspec.__dict__)

    @classmethod
    def from_config(cls, config):
        qspec = TFQuantizerSpec(num_bits=config["num_bits"],
                                mode=QuantizationMode.ASYMMETRIC,
                                signedness_to_force=None,
                                narrow_range=config["narrow_range"],
                                half_range=config['half_range'],
                                per_channel=config["per_channel"])
        return cls(qspec)
