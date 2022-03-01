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

from typing import Dict
from typing import Optional
from typing import List

import tensorflow as tf

from nncf.common.utils.registry import Registry
from nncf.common.quantization.structs import QuantizationMode
from nncf.tensorflow.layers.operation import InputType
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from nncf.tensorflow.quantization.quantizers import SymmetricQuantizer
from nncf.tensorflow.quantization.quantizers import AsymmetricQuantizer


NNCF_QUANTIZATION_OPERATIONS_V2 = Registry('nncf_quantization_operations_v2')


@NNCF_QUANTIZATION_OPERATIONS_V2.register(QuantizationMode.SYMMETRIC)
class SymmetricQuantizerV2(SymmetricQuantizer):
    def set_input_spec(self,
                       input_type: str,
                       input_shape: Optional[List[int]] = None,
                       channel_axes: Optional[List[int]] = None):
        """
        Sets input tensor specification for the quantizer.

        :param input_type: Indicates the type of input tensor: `inputs` or `weights`.
        :param input_shape: Shape of the input tensor for which the
            quantization is applied. Required only for per-channel
            quantization.
        :param channel_axes: Axes numbers of the input tensor which
            correspond to its channels. Required only for per-channel
            quantization.
        """
        self.input_type = input_type
        self.input_shape = input_shape
        self.channel_axes = channel_axes

    def create_variables(self, layer: tf.keras.layers.Layer) -> Dict[str, tf.Variable]:
        """
        Creates quantizer variables using `layer.add_weight()` method.

        :param layer: Instance of the `tf.keras.layers.Layer` class.
        :return: Quantizer variables.
        """
        if self.per_channel and (self.input_shape is None or self.channel_axes is None):
            raise ValueError('The `input_shape` and `channel_axes` arguments are required when'
                             'using per-channel quantization.')
        prefix = self.name
        return self._create_variables(layer, self.input_shape, self.channel_axes, prefix)


@NNCF_QUANTIZATION_OPERATIONS_V2.register(QuantizationMode.ASYMMETRIC)
class AsymmetricQuantizerV2(AsymmetricQuantizer):
    def set_input_spec(self,
                       input_type: str,
                       input_shape: Optional[List[int]] = None,
                       channel_axes: Optional[List[int]] = None):
        """
        Sets input tensor specification for the quantizer.

        :param input_type: Indicates the type of input tensor: `inputs` or `weights`.
        :param input_shape: Shape of the input tensor for which the
            quantization is applied. Required only for per-channel
            quantization.
        :param channel_axes: Axes numbers of the input tensor which
            correspond to its channels. Required only for per-channel
            quantization.
        """
        self.input_type = input_type
        self.input_shape = input_shape
        self.channel_axes = channel_axes

    def create_variables(self, layer: tf.keras.layers.Layer) -> Dict[str, tf.Variable]:
        """
        Creates quantizer variables using `layer.add_weight()` method.

        :param layer: Instance of the `tf.keras.layers.Layer` class.
        :return: Quantizer variables.
        """
        if self.per_channel and (self.input_shape is None or self.channel_axes is None):
            raise ValueError('The `input_shape` and `channel_axes` arguments are required when'
                             'using per-channel quantization.')
        prefix = self.name
        return self._create_variables(layer, self.input_shape, self.channel_axes, prefix)


def create_quantizer(name: str,
                     qspec: TFQuantizerSpec,
                     is_weight_quantization: bool,
                     input_shape: Optional[List[int]] = None,
                     channel_axes: Optional[List[int]] = None):
    """
    Factory method to create quantizer.

    :param name: Name of the quantizer. Should be unique.
    :param qspec: Specification of the quantizer.
    :param is_weight_quantization: A boolean flag.
        Takes one of the following values:
            - `True` if input tensor of the quantizer is weights
            - `False` if input tensor of the quantizer is activations
    :param input_shape: Shape of the input tensor for which the
        quantization is applied. Required only for per-channel
        quantization.
    :param channel_axes: Axes numbers of the input tensor which
        correspond to its channels. Required only for per-channel
        quantization.
    :return: The instance of the `SymmetricQuantizerV2` or
        `AsymmetricQuantizerV2` class.
    """
    quantizer_cls = NNCF_QUANTIZATION_OPERATIONS_V2.get(qspec.mode)
    input_type = InputType.WEIGHTS if is_weight_quantization else InputType.INPUTS
    quantizer = quantizer_cls(name, qspec)
    quantizer.set_input_spec(input_type, input_shape, channel_axes)
    return quantizer
