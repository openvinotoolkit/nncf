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

from typing import Optional
from dataclasses import dataclass
import numpy as np

from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic


@dataclass
class OVQuantizerLayerParameters:
    """
    Class handles FakeQuantize layer attributes.

    :param input_low: Tensor with minimum limit for input value.
    :param input_high: Tensor with maximum limit for input value.
    :param output_low: Tensor with minimum quantized value.
    :param output_high: Tensor with maximum quantized value.
    :param levels: Number of quantization levels.
    """
    input_low: np.ndarray
    input_high: np.ndarray
    output_low: np.ndarray
    output_high: np.ndarray
    levels: int


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig,
                                          axis: Optional[int]) -> OVQuantizerLayerParameters:
    """
    Calculates FakeQuantize layer attributes for weight quantizer.

    :param weight_tensor: Weight tensor to calculate quantizer attributes.
    :param quantizer_config: Config of FakeQuantize.
    :param axis: In per-channel case - the axis for the quantization. In per-tensor - ignored.
    :return: Parameters of the FakeQuantize layer.
    """
    if quantizer_config.per_channel:
        assert axis is not None
        axes = list(range(len(weight_tensor.shape)))
        axes.pop(axis)
        axes = tuple(axes)
    else:
        axes = None
    input_high = np.amax(weight_tensor, axis=axes, keepdims=True)
    input_low = np.amin(weight_tensor, axis=axes, keepdims=True)

    levels = 2 ** quantizer_config.num_bits
    if quantizer_config.mode == QuantizationMode.SYMMETRIC:
        output_low = np.full_like(input_low, fill_value=-levels / 2)
        output_high = np.full_like(input_high, fill_value=levels / 2 - 1)
    else:
        output_low = np.zeros_like(input_low)
        output_high = np.full_like(input_high, fill_value=levels - 1)
    return OVQuantizerLayerParameters(input_low, input_high, output_low, output_high, levels)


def calculate_activation_quantizer_parameters(statistics: MinMaxTensorStatistic,
                                              quantizer_config: QuantizerConfig) -> OVQuantizerLayerParameters:
    """
    Calculates FakeQuantize layer attributes for activation quantizer.

    :param statistics: Collected statistics for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :return: Parameters of the FakeQuantize layer.
    """
    input_low = np.array(statistics.min_values)
    input_high = np.array(statistics.max_values)
    levels = 2 ** quantizer_config.num_bits

    if quantizer_config.mode == QuantizationMode.SYMMETRIC:
        output_low = np.full_like(input_low, fill_value=-levels / 2)
        output_high = np.full_like(input_high, fill_value=levels / 2 - 1)
    else:
        output_low = np.zeros_like(input_low)
        output_high = np.full_like(input_high, fill_value=levels - 1)

    return OVQuantizerLayerParameters(input_low, input_high, output_low, output_high, levels)
