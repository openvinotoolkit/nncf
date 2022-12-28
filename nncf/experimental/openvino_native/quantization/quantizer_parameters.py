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

from typing import Optional, Tuple
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


def compute_levels(quantizer_config: QuantizerConfig, is_weights: bool) -> int:
    def_levels = 2 ** quantizer_config.num_bits

    if is_weights and quantizer_config.mode == QuantizationMode.SYMMETRIC:
        level_low = -def_levels / 2 + 1
    else:
        level_low = -def_levels / 2
    level_high = def_levels / 2 - 1
    return int(abs(level_high) + abs(level_low) + 1)


def fix_zero_filters_symmetric(max_level: np.ndarray, eps: float = 0.01) -> np.ndarray:
    max_range = np.max(max_level)
    lower_threshold = np.maximum(8e-5, eps * max_range)
    return np.maximum(lower_threshold, max_level)


def fix_zero_filters_asymmetric(max_level: np.ndarray, min_level: np.ndarray,
                                eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    ranges = max_level - min_level
    ranges = ranges if isinstance(ranges, np.ndarray) else np.array([ranges])
    min_correction = 8 * 10e-5
    corrections = [(np.maximum(eps * rng, rng) - rng) * 0.5 if rng > min_correction
                   else min_correction for rng in ranges]
    max_level = max_level + corrections
    min_level = min_level - corrections
    return max_level, min_level


def tune_range(left_border: np.ndarray, right_border: np.ndarray, num_bits: int) -> Tuple[np.ndarray, np.ndarray]:
    """ Tunes asymmetric quantization range to set zero quant precisely to zero value.
    Function moves left or right borders to do this and doesn't make left border higher or
    right border lesser than its original values
    :param left_border: range left border
    :param right_border: range right border
    :param num_bits: number of bits to perform quantization
    :return tuple with recomputed ranges
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
    max_level = fix_zero_filters_symmetric(max_values)
    if is_weights:
        min_level = -max_level
    else:
        signed = quantizer_config.signedness_to_force
        min_level = np.zeros(max_level.shape) if np.all(min_values >= 0) and not signed else \
            -max_level * levels / (levels - 2)

    return min_level, max_level


def asymmetric_range(min_values: np.ndarray, max_values: np.ndarray,
                     quantizer_config: QuantizerConfig) -> Tuple[np.ndarray, np.ndarray]:
    max_level, min_level = fix_zero_filters_asymmetric(max_values, min_values)
    min_level = np.where(min_level < 0.0, min_level, 0.0)
    max_level = np.where(max_level > 0.0, max_level, 0.0)
    min_level, max_level = tune_range(min_level, max_level, quantizer_config.num_bits)
    return min_level, max_level


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
    min_values = np.amin(weight_tensor, axis=axes, keepdims=True)
    max_values = np.amax(weight_tensor, axis=axes, keepdims=True)

    levels = compute_levels(quantizer_config, is_weights=True)
    if quantizer_config.mode == QuantizationMode.SYMMETRIC:
        min_level, max_level = symmetric_range(min_values, max_values, levels, quantizer_config, is_weights=True)
    else:
        min_level, max_level = asymmetric_range(min_values, max_values, quantizer_config)

    output_low, output_high = min_level, max_level
    return OVQuantizerLayerParameters(min_values, max_values, output_low, output_high, levels)


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
        min_level, max_level = symmetric_range(min_values, max_values, levels, quantizer_config, is_weights=False)
    else:
        min_level, max_level = asymmetric_range(min_values, max_values, quantizer_config)

    output_low, output_high = min_level, max_level
    return OVQuantizerLayerParameters(min_level, max_level, output_low, output_high, levels)
