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

from typing import Optional
from typing import List

import tensorflow as tf

from nncf.common.utils.registry import Registry
from nncf.common.quantization.structs import QuantizationMode
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.quantizers import Quantizer
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.functions import asymmetric_quantize
from nncf.tensorflow.quantization.functions import symmetric_quantize
from nncf.tensorflow.quantization.functions import asymmetric_range_initialization
from nncf.tensorflow.quantization.functions import symmetric_range_initialization


NNCF_QUANTIZATION_OPERATIONS_V2 = Registry('nncf_quantization_operations_v2')


def _get_channel_size(input_shape: List[int], channel_axes: List[int]):
    if not isinstance(channel_axes, (list, tuple)):
        channel_axes = [channel_axes]
    size = 1
    for axis in channel_axes:
        size *= input_shape[axis]
    return size


class QuantizerV2(Quantizer):
    """
    Base class for all quantization operations.
    """

    def __init__(self,
                 name: str,
                 qspec: TFQuantizerSpec,
                 input_type: str,
                 input_shape: Optional[List[int]] = None,
                 channel_axes: Optional[List[int]] = None):
        """
        Initializes the internal state of the quantizer.

        :param name: Name of operation. Unique identifier inside
            the NNCF network.
        :param qspec: Specification of the quantizer. Is a collection
            of parameters that influence how quantization performs.
        :param input_type: Indicates the type of input tensor: `inputs` or `weights`.
        :param input_shape: Shape of the input tensor for which the
            quantization is applied. Required only for per-channel
            quantization.
        :param channel_axes: Axes numbers of the input tensor which
            correspond to its channels. Required only for per-channel
            quantization.
        """
        super().__init__(name, qspec)

        # Specification of the input tensor
        self.input_type = input_type
        self._input_shape = input_shape
        self.channel_axes = channel_axes
        if self.per_channel and (self._input_shape is None or self.channel_axes is None):
            raise ValueError('The `input_shape` and `channel_axes` arguments are required when'
                             'using per-channel quantization.')

    @property
    def input_shape(self):
        return self._input_shape

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

    def quantize(self, inputs):
        """
        Applies quantization operation to the input tensor.

        :param inputs: Input tensor.
        :return: Quantized tensor.
        """
        raise NotImplementedError

    def apply_range_initialization(self, min_values, max_values, min_range=0.1, eps=0.01):
        """
        Initialize quantizer parameters using minimum and maximum weight values.

        :param min_values: Minimum weight values.
        :param max_values: Maximum weight values.
        :param min_range: Minimum range.
        :param eps: Smoothing coefficient for ranges: min_range = maximum(min_range, eps * max_range).
        """
        raise NotImplementedError

    def setup_input_transformation(self):
        self._pre_processing_fn, self._post_processing_fn = \
            QuantizerV2._make_transformation_fns(self._input_shape, self.channel_axes)


@NNCF_QUANTIZATION_OPERATIONS_V2.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizerV2(QuantizerV2):
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
        self._signedness_to_force = qspec.signedness_to_force
        # Following variables are initialized inside the `build()` method.
        self._scale_var = None  # type: tf.Variable
        self._signed_var = None  # type: tf.Variable

    @property
    def mode(self) -> str:
        return QuantizationMode.SYMMETRIC

    @property
    def signedness_to_force(self) -> Optional[bool]:
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
            self.setup_input_transformation()
            shape = (_get_channel_size(self._input_shape, self.channel_axes),)

        prefix = self.name.replace('/', '^')

        self._scale_var = nncf_network.add_weight(
            f'{prefix}^scale',
            shape=shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

        self._signed_var = nncf_network.add_weight(
            f'{prefix}^signed',
            initializer=tf.keras.initializers.Constant(
                -1.0 if self.signedness_to_force in (True, None) else 0.0
            ),
            trainable=False
        )

    def apply_overflow_fix(self):
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

    def apply_range_initialization(self, min_values, max_values, min_range=0.1, eps=0.01):
        signed, scale = symmetric_range_initialization(
            min_values, max_values, min_range, eps, self.signedness_to_force
        )
        self._signed_var.assign(signed)
        self._scale_var.assign(scale)


@NNCF_QUANTIZATION_OPERATIONS_V2.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizerV2(QuantizerV2):
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
    def signedness_to_force(self) -> Optional[bool]:
        return self._signedness_to_force

    def build(self, nncf_network) -> None:
        shape = None
        if self.per_channel:
            self.setup_input_transformation()
            shape = (_get_channel_size(self._input_shape, self.channel_axes),)

        prefix = self.name.replace('/', '^')

        self._input_low_var = nncf_network.add_weight(
            f'{prefix}^input_low',
            shape=shape,
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True
        )

        self._input_range_var = nncf_network.add_weight(
            f'{prefix}^input_range',
            shape=shape,
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )

    def apply_overflow_fix(self):
        if self.num_bits != 8 or not self._half_range:
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

    def apply_range_initialization(self, min_values, max_values, min_range=0.1, eps=0.01):
        input_low, input_range = asymmetric_range_initialization(
            min_values, max_values, min_range, eps
        )
        self._input_low_var.assign(input_low)
        self._input_range_var.assign(input_range)


def create_quantizer(name: str,
                     qspec: TFQuantizerSpec,
                     is_weight_quantization: bool,
                     input_shape: Optional[List[int]] = None,
                     channel_axes: Optional[List[int]] = None):
    quantizer_cls = NNCF_QUANTIZATION_OPERATIONS_V2.get(qspec.mode)
    input_type = InputType.WEIGHTS if is_weight_quantization else InputType.INPUTS
    return quantizer_cls(name, qspec, input_type, input_shape, channel_axes)
