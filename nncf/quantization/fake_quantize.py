# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from nncf.common.quantization.quantizers import calculate_asymmetric_level_ranges
from nncf.common.quantization.quantizers import calculate_symmetric_level_ranges
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic


@dataclass
class FakeQuantizeParameters:
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


def fix_zero_filters_asymmetric(
    min_values: np.ndarray, max_values: np.ndarray, eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray]:
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
    ranges = ranges.flatten() if isinstance(ranges, np.ndarray) else np.array([ranges])
    min_correction = 8e-4
    corrections = [
        (np.maximum(eps * rng, rng) - rng) * 0.5 if rng > min_correction else min_correction for rng in ranges
    ]
    corrections = np.array(corrections).reshape(max_values.shape)
    level_low = min_values - corrections
    level_high = max_values + corrections
    return level_low, level_high


def tune_range(
    left_border: np.ndarray, right_border: np.ndarray, num_bits: int, unify_zp: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tunes asymmetric quantization range to unify the zero point of all channels if `unify_zp` is True,
    or sets zero quant precisely to zero value otherwise.
    Function moves left or right borders to do this and doesn't make left border higher or
    right border lesser than its original values.

    :param left_border: Range left border.
    :param right_border: Range right border.
    :param num_bits: Number of bits to perform quantization.
    :param unify_zp: Whether to unify the zero point of all channels.
        If `True` calculates the average zero point of all channels and tune the max/min range.
    :return: Tuple with recomputed ranges.
    """
    level_high = 2**num_bits - 1

    if unify_zp:
        scale = (right_border - left_border) / level_high
        zero_point = -left_border / scale
        avg_zpts = np.round(np.mean(zero_point))
        qval = np.ones_like(left_border) * avg_zpts
    else:
        s = level_high / (right_border - left_border)
        fval = -left_border * s
        qval = np.round(fval)

    with np.errstate(invalid="ignore", divide="ignore"):
        ra = np.where(qval < level_high, qval / (qval - level_high) * right_border, left_border)
        rb = np.where(qval > 0.0, (qval - level_high) / qval * left_border, right_border)

    range_a = right_border - ra
    range_b = rb - left_border

    mask = np.where(range_a > range_b, 1.0, 0.0)
    inv_mask = np.abs(1.0 - mask)

    ra = mask * ra + inv_mask * left_border
    rb = inv_mask * rb + mask * right_border

    return ra, rb


def symmetric_range(
    min_values: np.ndarray,
    max_values: np.ndarray,
    levels: int,
    quantizer_config: QuantizerConfig,
    q_group: QuantizerGroup,
) -> Tuple[np.ndarray, np.ndarray]:
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
    if q_group == QuantizerGroup.WEIGHTS:
        level_low = -level_high
    else:
        signed = quantizer_config.signedness_to_force is True
        level_low = (
            np.zeros_like(level_high) if np.all(min_values >= 0) and not signed else -level_high * levels / (levels - 2)
        )

    level_low = level_low.astype(np.float32)
    level_high = level_high.astype(np.float32)
    return level_low, level_high


def asymmetric_range(
    min_values: np.ndarray,
    max_values: np.ndarray,
    quantizer_config: QuantizerConfig,
    q_group: QuantizerGroup,
    unify_zp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the numbers of the low and high quant for the asymmetric quantization scheme.

    :param min_values: Collected min values for the quantized insertion.
    :param max_values: Collected max values for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :param unify_zp: Whether to unify the zero point.
        It is `True` for the per-tensor zero point constrain on KMB (vpu2p0).
    :return: A Tuple
        level_low - the low quant number
        level_high - the high quant number
    """
    level_low, level_high = fix_zero_filters_asymmetric(min_values, max_values)
    level_low = np.where(level_low < 0.0, level_low, 0.0)
    level_high = np.where(level_high > 0.0, level_high, 0.0)

    if unify_zp and q_group == QuantizerGroup.ACTIVATIONS:
        raise NotImplementedError("Unified zero point is not supported for activations.")

    level_low, level_high = tune_range(level_low, level_high, quantizer_config.num_bits, unify_zp=unify_zp)
    level_low = level_low.astype(np.float32)
    level_high = level_high.astype(np.float32)
    return level_low, level_high


def get_quantizer_narrow_range(quantizer_config: QuantizerConfig, quant_group: QuantizerGroup) -> bool:
    """
    Returns narrow_range parameter: True if the range of quantized values is reduced by 1 compared to the
        naive case, False otherwise.

    :param quantizer_config: Config of the quantization configuration.
    :param quant_group: Group of the quantizer.
    :return: narrow_range parameter.
    """
    if quantizer_config.mode == QuantizationMode.SYMMETRIC:
        return quant_group == QuantizerGroup.WEIGHTS
    return False


def calculate_quantizer_parameters(
    statistics: MinMaxTensorStatistic,
    quantizer_config: QuantizerConfig,
    quant_group: QuantizerGroup,
    narrow_range: bool,
    half_range: bool = False,
) -> FakeQuantizeParameters:
    """
    Calculates FakeQuantize layer attributes for weight/activation quantizer.

    :param statistics: Collected statistics for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :param quantizer_group: Group of the quantizer.
    :param narrow_range: True if the range of quantized values is reduced by 1 compared to the
        naive case, False otherwise.
    :param half_range: If True effectively only a half of a quantizer range is used.
        False - the full range is used.
    :return: Parameters of the FakeQuantize layer.
    """
    min_values = np.array(statistics.min_values).astype(np.float32)
    max_values = np.array(statistics.max_values).astype(np.float32)

    if half_range:
        input_low, input_high, levels = _calculate_scaled_parameters(
            min_values, max_values, quantizer_config, quant_group, narrow_range
        )
    else:
        num_bits = quantizer_config.num_bits
        if quantizer_config.mode == QuantizationMode.SYMMETRIC:
            _, _, levels = calculate_symmetric_level_ranges(num_bits, signed=True, narrow_range=narrow_range)
            input_low, input_high = symmetric_range(min_values, max_values, levels, quantizer_config, quant_group)
        else:
            _, _, levels = calculate_asymmetric_level_ranges(num_bits, narrow_range=narrow_range)
            input_low, input_high = asymmetric_range(min_values, max_values, quantizer_config, quant_group)

    if not quantizer_config.per_channel:
        input_low = np.squeeze(input_low)
        input_high = np.squeeze(input_high)

    input_low, input_high = np.array(input_low), np.array(input_high)
    output_low, output_high = input_low, input_high
    return FakeQuantizeParameters(input_low, input_high, output_low, output_high, levels)


def _calculate_scaled_parameters(
    min_values: np.ndarray,
    max_values: np.ndarray,
    quantizer_config: QuantizerConfig,
    quant_group: QuantizerGroup,
    narrow_range: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Calculates FakeQuantize layer attributes scaled to effectively use a half range of the quantization range.

    :param min_values: Minimum values of statistics for the quantizer.
    :param max_values: Maximum values of statistics for the quantizer.
    :param quantizer_config: Config of the quantization configuration.
    :param quantizer_group: Group of the quantizer.
    :param narrow_range: True if the range of quantized values is reduced by 1 compared to the
        naive case, False otherwise.
    :return: A Tuple
        input_low: Tensor with minimum limit for input value.
        input_high: Tensor with maximum limit for input value.
        levels: Number of quantization levels.
    """
    if quantizer_config.mode == QuantizationMode.ASYMMETRIC:
        raise RuntimeError("half_range is only applied to symmetric quantization mode.")
    if quant_group != QuantizerGroup.WEIGHTS:
        raise RuntimeError("half_range is only applied to weight quantizers.")

    num_bits = quantizer_config.num_bits
    _, _, levels = calculate_symmetric_level_ranges(num_bits - 1, signed=True, narrow_range=False)
    input_low, input_high = symmetric_range(min_values, max_values, levels, quantizer_config, quant_group)

    _, _, export_levels = calculate_symmetric_level_ranges(num_bits, signed=True, narrow_range=narrow_range)
    input_high *= (export_levels - 1) / (levels - 1)
    input_low *= (export_levels - 1) / (levels - 1)

    return input_low, input_high, export_levels
