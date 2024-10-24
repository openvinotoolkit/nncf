# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import nncf
from nncf.common.quantization.quantizers import calculate_asymmetric_level_ranges
from nncf.common.quantization.quantizers import calculate_symmetric_level_ranges
from nncf.common.quantization.quantizers import get_num_levels
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.quantization.advanced_parameters import FP8Type
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns


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

    input_low: Tensor
    input_high: Tensor
    output_low: Tensor
    output_high: Tensor
    levels: int


@dataclass
class FakeConvertParameters:
    """
    Class handles FakeConvert layer attributes.

    :param scale: Tensor with the scale for input value.
    :param shift: Tensor with the shift for input value.
    :param destination_type: Destination type.
    """

    scale: Tensor
    shift: Tensor
    destination_type: FP8Type


def fix_zero_filters_symmetric(max_values: Tensor, eps: float = 0.01) -> Tensor:
    """
    Fixes zero filters for symmetric quantizer.

    :param max_values: Collected max values for the quantized insertion.
    :param eps: Correction coefficient.
    :return: Fixed the high quant number.
    """
    max_range = fns.max(max_values)
    lower_threshold = fns.maximum(max_range * eps, 8e-5)
    return fns.maximum(lower_threshold, max_values)


def fix_zero_filters_asymmetric(min_values: Tensor, max_values: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
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
    min_correction = 8e-4
    corrections = fns.where(ranges > min_correction, (fns.maximum(eps * ranges, ranges) - ranges) * 0.5, min_correction)

    level_low = min_values - corrections
    level_high = max_values + corrections
    return level_low, level_high


def tune_range(
    left_border: Tensor, right_border: Tensor, num_bits: int, unify_zp: bool = False
) -> Tuple[Tensor, Tensor]:
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
        avg_zpts = fns.round(fns.mean(zero_point))
        qval = fns.ones_like(left_border) * avg_zpts
    else:
        s = level_high / (right_border - left_border)
        fval = -left_border * s
        qval = fns.round(fval)

    ra = fns.where(qval < level_high, qval / (qval - level_high) * right_border, left_border)
    with warnings.catch_warnings():
        # If `qval` is 0 `rb` will equal `right_border`, and we don't want to show an unnecessary division by 0 warning
        warnings.simplefilter("ignore")
        rb_then_result = (qval - level_high) / qval * left_border
    rb = fns.where(qval > 0.0, rb_then_result, right_border)

    range_a = right_border - ra
    range_b = rb - left_border

    mask = fns.where(range_a > range_b, 1.0, 0.0)
    inv_mask = fns.abs(1.0 - mask)

    ra = mask * ra + inv_mask * left_border
    rb = inv_mask * rb + mask * right_border

    return ra, rb


def symmetric_range(
    min_values: Tensor,
    max_values: Tensor,
    levels: int,
    quantizer_config: QuantizerConfig,
    q_group: QuantizerGroup,
) -> Tuple[Tensor, Tensor]:
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
            fns.zeros_like(level_high)
            if fns.all(min_values >= 0) and not signed
            else -level_high * levels / (levels - 2)
        )

    level_low = level_low.astype(TensorDataType.float32)
    level_high = level_high.astype(TensorDataType.float32)
    return level_low, level_high


def asymmetric_range(
    min_values: Tensor,
    max_values: Tensor,
    quantizer_config: QuantizerConfig,
    q_group: QuantizerGroup,
    unify_zp: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Calculates the numbers of the low and high quant for the asymmetric quantization scheme.

    :param min_values: Collected min values for the quantized insertion.
    :param max_values: Collected max values for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :param unify_zp: Whether to unify the zero point.
        It is `True` for the per-tensor zero point constrain on KMB.
    :return: A Tuple
        level_low - the low quant number
        level_high - the high quant number
    """
    level_low, level_high = fix_zero_filters_asymmetric(min_values, max_values)
    level_low = fns.where(level_low < 0.0, level_low, 0.0)
    level_high = fns.where(level_high > 0.0, level_high, 0.0)

    if unify_zp and q_group == QuantizerGroup.ACTIVATIONS:
        raise NotImplementedError("Unified zero point is not supported for activations.")

    level_low, level_high = tune_range(level_low, level_high, quantizer_config.num_bits, unify_zp=unify_zp)
    level_low = level_low.astype(TensorDataType.float32)
    level_high = level_high.astype(TensorDataType.float32)
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
    min_values = statistics.min_values.astype(TensorDataType.float32)
    max_values = statistics.max_values.astype(TensorDataType.float32)

    if half_range:
        input_low, input_high, levels = _calculate_scaled_parameters(
            min_values, max_values, quantizer_config, quant_group, narrow_range
        )
    else:
        num_bits = quantizer_config.num_bits
        if quantizer_config.mode == QuantizationMode.SYMMETRIC:
            level_low, level_high = calculate_symmetric_level_ranges(num_bits, signed=True, narrow_range=narrow_range)
            levels = get_num_levels(level_low, level_high)
            input_low, input_high = symmetric_range(min_values, max_values, levels, quantizer_config, quant_group)
        else:
            level_low, level_high = calculate_asymmetric_level_ranges(num_bits, narrow_range=narrow_range)
            levels = get_num_levels(level_low, level_high)
            input_low, input_high = asymmetric_range(min_values, max_values, quantizer_config, quant_group)

    if not quantizer_config.per_channel:
        input_low = fns.squeeze(input_low)
        input_high = fns.squeeze(input_high)

    output_low, output_high = input_low, input_high
    return FakeQuantizeParameters(input_low, input_high, output_low, output_high, levels)


def calculate_convert_parameters(
    statistics: MinMaxTensorStatistic,
    is_per_channel: False,
    destination_type: FP8Type = FP8Type.E4M3,
    activation_scale: float = 0.5,
) -> FakeConvertParameters:
    """
    Calculates FakeConvert layer attributes for weight/activation quantizer.

    :param statistics: Collected statistics for the quantized insertion.
    :param is_activation: Whether is for activation or weights.
    :param destination_type: Destination type that regulates maximum value for the formula.
    :param activation_scale: Factor for calculated activation scale.
    :return: Parameters of the FakeConvert layer.
    """

    destination_type_maximum = {FP8Type.E4M3: 448, FP8Type.E5M2: 57344}

    max_values = statistics.max_values
    min_values = statistics.min_values

    max_destination_value = destination_type_maximum[destination_type]
    tensor_dtype = fns.finfo(max_values)
    scale = max_destination_value / fns.maximum(max_values, fns.abs(min_values) + tensor_dtype.eps)
    if not is_per_channel:
        scale = fns.squeeze(activation_scale * scale)
    shift = fns.zeros_like(scale).astype(TensorDataType.float32)
    scale = scale.astype(TensorDataType.float32)
    return FakeConvertParameters(scale, shift, destination_type)


def _calculate_scaled_parameters(
    min_values: Tensor,
    max_values: Tensor,
    quantizer_config: QuantizerConfig,
    quant_group: QuantizerGroup,
    narrow_range: bool,
) -> Tuple[Tensor, Tensor, int]:
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
        raise nncf.ValidationError("half_range is only applied to symmetric quantization mode.")
    if quant_group != QuantizerGroup.WEIGHTS:
        raise nncf.ValidationError("half_range is only applied to weight quantizers.")

    num_bits = quantizer_config.num_bits
    level_low, level_high = calculate_symmetric_level_ranges(num_bits - 1, signed=True, narrow_range=False)
    levels = get_num_levels(level_low, level_high)
    input_low, input_high = symmetric_range(min_values, max_values, levels, quantizer_config, quant_group)

    export_level_low, export_level_high = calculate_symmetric_level_ranges(
        num_bits, signed=True, narrow_range=narrow_range
    )
    export_levels = get_num_levels(export_level_low, export_level_high)
    input_high *= (export_levels - 1) / (levels - 1)
    input_low *= (export_levels - 1) / (levels - 1)

    return input_low, input_high, export_levels


def calculate_scale_zero_point(
    input_low: Tensor,
    input_high: Tensor,
    level_low: int,
    level_high: int,
    narrow_range: bool,
    invert_division: Optional[bool] = False,
) -> Tuple[Tensor, Tensor]:
    """
    Calculates scale and zero_point values for the quantizer.

    :param input_low: The minimum limit for an input value based on collected statistics.
    :param input_high: The maximum limit for an input value based on collected statistics.
    :param level_low: The minimum level in the integer range to quantize.
        The default is "0" for an unsigned range, and "-2^(bit-1)" for a signed one .
    :param level_high: The maximum level in the integer range to quantize.
        The default is "2^bits-1" for an unsigned range, and "2^(bit-1)-1" for a signed one.
    :param narrow_range: True if the range of quantized values is narrowed as compared to the
        naive case, False otherwise.
    :return: Scale and Zero point values.
    """
    levels = level_high - level_low if narrow_range else level_high - level_low + 1
    if invert_division:
        scale = ((input_high - input_low) * (1.0 / (levels - 1))).astype(TensorDataType.float32)
    else:
        scale = ((input_high - input_low) / (levels - 1)).astype(TensorDataType.float32)
    eps = fns.finfo(scale).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale = fns.where(fns.abs(scale) < eps, eps, scale)
    expected_level_low = level_low + 1 if narrow_range else level_low
    if invert_division:
        zero_point = expected_level_low - fns.round(input_low * (1.0 / scale))
    else:
        zero_point = expected_level_low - fns.round(input_low / scale)
    zero_point = fns.clip(zero_point.astype(TensorDataType.int32), level_low, level_high)
    return scale, zero_point
