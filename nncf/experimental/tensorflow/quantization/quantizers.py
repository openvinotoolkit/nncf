"""
 Copyright (c) 2021 Intel Corporation
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
from typing import List
from abc import abstractmethod
from enum import Enum

import tensorflow as tf

from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerSpec
from nncf.common.utils.registry import Registry
from nncf.tensorflow.quantization.functions import asymmetric_quantize
from nncf.tensorflow.quantization.functions import symmetric_quantize
from nncf.experimental.tensorflow.nncf_operation import NNCFOperation


NNCF_QUANTIZATION_OPERATIONS = Registry('nncf_quantization_operations')


def _get_channel_size(input_shape: List[int], channel_axes: List[int]):
    if not isinstance(channel_axes, (list, tuple)):
        channel_axes = [channel_axes]
    size = 1
    for axis in channel_axes:
        size *= input_shape[axis]
    return size


class TFQuantizerSpec(QuantizerSpec):
    def __init__(self,
                 num_bits: int,
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


class InputType(Enum):
    INPUTS = 'inputs'
    WEIGHTS = 'weights'

    @classmethod
    def from_str(cls, value: str) -> 'InputType':
        return cls(value)


class Quantizer(NNCFOperation):
    """
    Base class for all quantization operations.
    """

    def __init__(self,
                 name: str,
                 qspec: TFQuantizerSpec,
                 input_type: InputType,
                 input_shape: Optional[List[int]] = None,
                 channel_axes: Optional[List[int]] = None):
        """
        Initializes the internal state of the quantizer.

        :param name: Name of operation. Unique identifier inside
            the NNCF network.
        :param qspec: Specification of the quantizer. Is a collection
            of parameters that influence how quantization performs.
        :param input_type:
        :param input_shape: Shape of the input tensor for which the
            quantization is applied. Required only for per-channel
            quantization.
        :param channel_axes: Axes numbers of the input tensor which
            correspond to its channels. Required only for per-channel
            quantization.
        """
        super().__init__(name)

        # Specification of the quantization
        self.num_bits = qspec.num_bits
        self.per_channel = qspec.per_channel
        self.narrow_range = qspec.narrow_range
        self.half_range = qspec.half_range

        # Specification of the input tensor
        self.input_type = input_type
        self._input_shape = input_shape
        self._channel_axes = channel_axes
        if self.per_channel and (self._input_shape is None or self._channel_axes is None):
            raise ValueError('The `input_shape` and `channel_axes` arguments are required when'
                             'using per-channel quantization.')

        self._enabled = True

        self._eps = 1e-16
        self._pre_processing_fn = self._make_pre_processing_fn()
        self._post_processing_fn = self._make_post_processing_fn()

    @property
    def channel_axes(self):
        return self._channel_axes

    @property
    @abstractmethod
    def mode(self) -> str:
        """
        Returns mode of the quantization (symmetric or asymmetric).

        :return: The mode of the quantization.
        """

    @property
    @abstractmethod
    def signedness_to_force(self) -> bool:
        """
        """

    @property
    def enabled(self) -> bool:
        """
        Returns the boolean flag that specified whether the quantization
        operation applying or not.

        :return: Boolean flag.
        """
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    def call(self, inputs, *args, **kwargs):
        """
        Applies quantization to the input tensor if the quantizer is enabled.
        Otherwise, if the quantizer is disabled, returns the input tensor as-is.

        :param inputs: Input tensor.
        :return: Output tensor.
        """
        if not self.enabled:
            return inputs
        transformed = self._pre_processing_fn(inputs)
        quantized = self.quantize(transformed)
        outputs = self._post_processing_fn(quantized)
        return outputs

    @abstractmethod
    def quantize(self, inputs):
        """
        Applies quantization operation to the input tensor.

        :param inputs: Input tensor.
        :return: Quantized tensor.
        """

    @abstractmethod
    def apply_range_initialization(self, min_values, max_values, min_range: float = 0.1, eps: float = 0.01) -> None:
        """
        Initializes quantizer parameters using minimum and maximum weight values.

        :param min_values: Minimum weight values.
        :param max_values: Maximum weight values.
        :param min_range: Minimum range.
        :param eps: Smoothing coefficient for ranges: min_range = maximum(min_range, eps * max_range).
        """

    def get_quantizer_config(self) -> QuantizerConfig:
        """
        Used to get a current quantizer state in terms of QuantizerConfig objects.

        :return: A QuantizerConfig struct that corresponds to current state of the quantizer.
        """
        return QuantizerConfig(self.num_bits, self.mode, self.signedness_to_force, self.per_channel)

    def get_config(self) -> Dict[str, Any]:
        config = {
            'name': self.name,
            'input_type': self.input_type.value,
            'quantizer_spec': {
                'num_bits': self.num_bits,
                'mode': self.mode,
                'signedness_to_force': self.signedness_to_force,
                'narrow_range': self.narrow_range,
                'half_range': self.half_range,
                'per_channel': self.per_channel,
            }
        }
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        qspec = TFQuantizerSpec.from_state(config['quantizer_spec'])
        input_type = InputType.from_str(config['input_type'])
        name = config['name']
        return cls(name, qspec, input_type)

    def _setup_input_transformation(self):
        """
        Setup input transformation that the per-channel quantization can be applied to input tensor.
        The TensorFlow fake_quant_with_min_max_vars_per_channel supports only inputs tensor one of
        the shapes: [d], [b, d] [b, h, w, d]. For this reason, Quantizer transforms any inputs tensor
        to one of the supported shapes, then quantizes and then transforms quantized tensor to
        the original inputs shape.
        """
        self._pre_processing_fn, self._post_processing_fn = Quantizer._make_transformation_fns(self._input_shape,
                                                                                               self._channel_axes)

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
                raise NotImplementedError('Quntizer could not transform input to apply per-channel quantization: '
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


@NNCF_QUANTIZATION_OPERATIONS.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizer(Quantizer):
    """
    Represents the nncf operation that performs symmetric quantization.
    """

    def __init__(self,
                 name: str,
                 qspec: TFQuantizerSpec,
                 input_type: InputType,
                 input_shape: Optional[List[int]] = None,
                 channel_axes: Optional[List[int]] = None):
        """
        Initializes the internal state of the symmetric quantizer.
        """
        super().__init__(name, qspec, input_type, input_shape, channel_axes)
        # Specification of the quantization
        self._signedness_to_force = qspec.signedness_to_force
        # Following variables are initialized inside the `build()` method.
        self._scale_var = None  # type: tf.Variable
        self._signed_var = None  # type: tf.Variable

    @property
    def mode(self) -> str:
        return QuantizationMode.SYMMETRIC

    @property
    def signedness_to_force(self) -> bool:
        return self._signedness_to_force

    @property
    def signed(self) -> bool:
        """
        Returns `True` for signed quantization, `False` for unsigned.

        :return: `True` for signed quantization, `False` for unsigned.
        """
        return self._signed_var.numpy() < 0.0

    def build(self, nncf_network) -> None:
        shape = None
        if self.per_channel:
            self._setup_input_transformation()
            shape = (_get_channel_size(self._input_shape, self._channel_axes),)

        # TODO(andrey-churkin):
        with tf.name_scope(self.name):
            self._scale_var = nncf_network.add_weight(
                'scale',
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )

            self._signed_var = nncf_network.add_weight(
                'signed',
                initializer=tf.keras.initializers.Constant(
                    -1.0 if self.signedness_to_force in (True, None) else 0.0
                ),
                trainable=False
            )

    def apply_saturation_fix(self):
        if self.num_bits != 8 or not self.half_range:
            raise RuntimeError('Attempt to apply saturation issue fix '
                               'to quantizer which is not configured for that.')

        # Multiplier to expand scale from 7 bit to 8 bit
        multiplier = 127 / 63 if self.narrow_range else 255 / 127
        self._scale_var.assign(multiplier * self._scale_var)
        self._eps *= multiplier
        self._half_range = False

    def quantize(self, inputs):
        num_bits = self.num_bits - 1 if self.half_range else self.num_bits
        return symmetric_quantize(
            inputs,
            self._scale_var,
            self._signed_var,
            num_bits,
            self.per_channel,
            self.narrow_range,
            self._eps
        )

    def apply_range_initialization(self, min_values, max_values, min_range: float = 0.1, eps: float = 0.01):
        if self.signedness_to_force is None:
            sign = tf.reduce_any(tf.less(min_values, 0))
            self._signed_var.assign(-1.0 if sign else 0.0)
        ranges = tf.maximum(tf.abs(max_values), tf.abs(min_values))
        max_range = tf.reduce_max(ranges)
        lower_threshold = tf.maximum(eps * max_range, min_range)
        scale = tf.maximum(ranges, lower_threshold)
        self._scale_var.assign(scale)


@NNCF_QUANTIZATION_OPERATIONS.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizer(Quantizer):
    """
    Represents the nncf operation that performs asymmetric quantization.
    """

    def __init__(self,
                 name: str,
                 qspec: TFQuantizerSpec,
                 input_type: InputType,
                 input_shape: Optional[List[int]] = None,
                 channel_axes: Optional[List[int]] = None):
        """
        Initializes the internal state of the symmetric quantizer.
        """
        super().__init__(name, qspec, input_type, input_shape, channel_axes)
        # Specification of the quantization
        self._signedness_to_force = None
        # Following variables are initialized inside the `build()` method.
        self._input_low_var = None  # type: tf.Variable
        self._input_range_var = None  # type: tf.Variable

    @property
    def mode(self) -> str:
        return QuantizationMode.ASYMMETRIC

    @property
    def signedness_to_force(self) -> None:
        return self._signedness_to_force

    def build(self, nncf_network) -> None:
        shape = None
        if self.per_channel:
            self._setup_input_transformation()
            shape = (_get_channel_size(self._input_shape, self._channel_axes),)

        # TODO(andrey-churkin):
        with tf.name_scope(self.name):
            self._input_low_var = nncf_network.add_weight(
                'input_low',
                shape=shape,
                initializer=tf.keras.initializers.Constant(0.0),
                trainable=True
            )

            self._input_range_var = nncf_network.add_weight(
                'input_range',
                shape=shape,
                initializer=tf.keras.initializers.Constant(1.0),
                trainable=True
            )

    def apply_saturation_fix(self):
        if self.num_bits != 8 or not self.half_range:
            raise RuntimeError('Attempt to apply saturation issue fix '
                               'to quantizer which is not configured for that.')

        # Low value shift to expand quantize range from 7 bit to 8 bit properly
        self._input_low_var.assign(
            self._input_low_var + self._min_adj(7, self._input_low_var, self._input_range_var + self._eps,
                                                self.narrow_range)
        )

        # Multiplier to expand scale from 7 bit to 8 bit
        multiplier = 127 / 63 if self.narrow_range else 255 / 127
        self._input_range_var.assign(multiplier * self._input_range_var)
        self._eps *= multiplier
        self._half_range = False

    def quantize(self, inputs):
        num_bits = self.num_bits - 1 if self.half_range else self.num_bits
        return asymmetric_quantize(
            inputs,
            self._input_low_var,
            self._input_range_var,
            num_bits,
            self.per_channel,
            self.narrow_range,
            self._eps
        )

    def apply_range_initialization(self, min_values, max_values, min_range: float = 0.1, eps: float = 0.01):
        ranges = max_values - min_values
        max_range = tf.reduce_max(ranges)
        lower_threshold = tf.maximum(eps * max_range, min_range)
        correction = (tf.maximum(ranges, lower_threshold) - ranges) * 0.5
        input_low = min_values - correction
        input_range = ranges + 2 * correction
        self._input_low_var.assign(input_low)
        self._input_range_var.assign(input_range)


def create_quantizer(name: str,
                     qspec: TFQuantizerSpec,
                     is_weight_quantization: bool,
                     input_shape: Optional[List[int]] = None,
                     channel_axes: Optional[List[int]] = None):
    """
    :param name:
    :param qspec:
    :param is_weight_quantization:
    :param input_shape:
    :param channel_axes:
    :return:
    """
    quantizer_cls = NNCF_QUANTIZATION_OPERATIONS.get(qspec.mode)
    input_type = InputType.WEIGHTS if is_weight_quantization else InputType.INPUTS
    return quantizer_cls(name, qspec, input_type, input_shape, channel_axes)
