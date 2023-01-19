"""
 Copyright (c) 2023 Intel Corporation
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

from typing import List, Tuple, Type
from dataclasses import dataclass

import numpy as np

from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.openvino_native.hardware.pattern_operations import TRANSPOSED_OPERATIONS


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


def compute_levels(quantizer_config: QuantizerConfig, is_weights: bool) -> int:
    """
    Calculates number of quantization levels.

    :param quantizer_config: Config of the quantization configuration.
    :param is_weights: `True` for weight quantizer and `False` otherwise.
    :return: Number of quantization levels.
    """
    def_levels = 2 ** quantizer_config.num_bits

    if is_weights and quantizer_config.mode == QuantizationMode.SYMMETRIC:
        level_low = -def_levels / 2 + 1
    else:
        level_low = -def_levels / 2
    level_high = def_levels / 2 - 1
    return int(abs(level_high) + abs(level_low) + 1)


def fix_zero_filters_symmetric(max_values: np.ndarray, eps: float = 0.01) -> np.ndarray:
    """
    Fixes zero filters for symmetric quantizer.

    :param max_values: Collected max values for the quantized insertion.
    :param eps: Correction coefficient.
    :return: Fixed the high quant number.
    """
    max_range = np.max(max_values)
    lower_threshold = np.maximum(8e-5, eps * max_range)
    return np.maximum(lower_threshold, max_values)


def fix_zero_filters_asymmetric(min_values: np.ndarray, max_values: np.ndarray,
                                eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fixes zero filters for asymmetric quantizer.

    :param min_values: Collected min values for the quantized insertion.
    :param max_values: Collected max values for the quantized insertion.
    :param eps: Correction coefficient.
    :return: A Tuple
        level_low - fixed the low quant number
        level_high - fixed the high quant number
    """
    ranges = max_values - min_values
    ranges = ranges if isinstance(ranges, np.ndarray) else np.array([ranges])
    min_correction = 8e-4
    corrections = [(np.maximum(eps * rng, rng) - rng) * 0.5 if rng > min_correction
                   else min_correction for rng in ranges]
    level_low = min_values - corrections
    level_high = max_values + corrections
    return level_low, level_high


def tune_range(left_border: np.ndarray, right_border: np.ndarray, num_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tunes asymmetric quantization range to set zero quant precisely to zero value.
    Function moves left or right borders to do this and doesn't make left border higher or
    right border lesser than its original values.

    :param left_border: Range left border.
    :param right_border: Range right border.
    :param num_bits: Number of bits to perform quantization.
    :return: Tuple with recomputed ranges.
    """
    level_high = 2 ** num_bits - 1
    s = level_high / (right_border - left_border)
    fval = -left_border * s
    qval = np.round(fval)

    with np.errstate(invalid='ignore', divide='ignore'):
        ra = np.where(qval < level_high, qval / (qval - level_high) * right_border, left_border)
        rb = np.where(qval > 0.0, (qval - level_high) / qval * left_border, right_border)

    range_a = right_border - ra
    range_b = rb - left_border

    mask = np.where(range_a > range_b, 1.0, 0.0)
    inv_mask = np.abs(1.0 - mask)

    ra = mask * ra + inv_mask * left_border
    rb = inv_mask * rb + mask * right_border

    return ra, rb


def symmetric_range(min_values: np.ndarray, max_values: np.ndarray, levels: int,
                    quantizer_config: QuantizerConfig, is_weights: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the numbers of the low and high quant for the symmetric quantization scheme.

    :param min_values: Collected min values for the quantized insertion.
    :param max_values: Collected max values for the quantized insertion.
    :param levels: Number of quantization levels.
    :param quantizer_config: Config of the quantization configuration.
    :return: A Tuple
        level_low - the low quant number
        level_high - the high quant number
    """
    level_high = fix_zero_filters_symmetric(max_values)
    if is_weights:
        level_low = -level_high
    else:
        signed = quantizer_config.signedness_to_force
        level_low = np.zeros(level_high.shape) if np.all(min_values >= 0) and not signed else \
            -level_high * levels / (levels - 2)

    level_low = level_low.astype(np.float32)
    level_high = level_high.astype(np.float32)
    return level_low, level_high


def asymmetric_range(min_values: np.ndarray, max_values: np.ndarray,
                     quantizer_config: QuantizerConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the numbers of the low and high quant for the asymmetric quantization scheme.

    :param min_values: Collected min values for the quantized insertion.
    :param max_values: Collected max values for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :return: A Tuple
        level_low - the low quant number
        level_high - the high quant number
    """
    level_low, level_high = fix_zero_filters_asymmetric(min_values, max_values)
    level_low = np.where(level_low < 0.0, level_low, 0.0)
    level_high = np.where(level_high > 0.0, level_high, 0.0)
    level_low, level_high = tune_range(level_low, level_high, quantizer_config.num_bits)

    level_low = level_low.astype(np.float32)
    level_high = level_high.astype(np.float32)
    return level_low, level_high


def get_weight_stats_shape(const_shape: List[int], metatype: Type[OperatorMetatype]) -> List[int]:
    """
    Calculates shapes for FakeQuantize statistics.

    :param const_shape: Shape of the weight tensor.
    :param metatype: NNCF meta type which corresponds to operation.
    :return: Shapes for FakeQuantize statistics.
    """
    bounds_shape = np.ones(len(const_shape), dtype=np.int32)
    if metatype in TRANSPOSED_OPERATIONS:
        bounds_shape[1] = const_shape[1]
    else:
        bounds_shape[0] = const_shape[0]
    return bounds_shape


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig,
                                          metatype: Type[OperatorMetatype]) -> OVQuantizerLayerParameters:
    """
    Calculates FakeQuantize layer attributes for weight quantizer.

    :param weight_tensor: Weight tensor to calculate quantizer attributes.
    :param quantizer_config: Config of FakeQuantize.
    :param axis: In per-channel case - the axis for the quantization. In per-tensor - ignored.
    :return: Parameters of the FakeQuantize layer.
    """
    axes = []
    bounds_shape = get_weight_stats_shape(weight_tensor.shape, metatype)
    for i, dim in enumerate(bounds_shape):
        if dim == 1:
            axes.append(i)
    axes = tuple(axes)

    min_values = np.amin(weight_tensor, axis=axes, keepdims=True)
    max_values = np.amax(weight_tensor, axis=axes, keepdims=True)

    levels = compute_levels(quantizer_config, is_weights=True)
    if quantizer_config.mode == QuantizationMode.SYMMETRIC:
        level_low, level_high = symmetric_range(min_values, max_values, levels, quantizer_config, is_weights=True)
    else:
        level_low, level_high = asymmetric_range(min_values, max_values, quantizer_config)

    output_low, output_high = level_low, level_high
    return OVQuantizerLayerParameters(level_low, level_high, output_low, output_high, levels)


def calculate_activation_quantizer_parameters(statistics: MinMaxTensorStatistic,
                                              quantizer_config: QuantizerConfig) -> OVQuantizerLayerParameters:
    """
    Calculates FakeQuantize layer attributes for activation quantizer.

    :param statistics: Collected statistics for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :return: Parameters of the FakeQuantize layer.
    """
    levels = compute_levels(quantizer_config, is_weights=False)
    min_values = np.array(statistics.min_values)
    max_values = np.array(statistics.max_values)

    if quantizer_config.mode == QuantizationMode.SYMMETRIC:
        level_low, level_high = symmetric_range(min_values, max_values, levels, quantizer_config, is_weights=False)
    else:
        level_low, level_high = asymmetric_range(min_values, max_values, quantizer_config)

    output_low, output_high = level_low, level_high
    return OVQuantizerLayerParameters(level_low, level_high, output_low, output_high, levels)
