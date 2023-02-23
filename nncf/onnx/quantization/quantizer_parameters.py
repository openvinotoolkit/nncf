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

from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np
from nncf.common.quantization.structs import QuantizationMode
from nncf.quantization.fake_quantize import FakeQuantizeParameters


@dataclass
class ONNXQuantizerLayerParameters:
    """
    Class handles Quantizer layer attributes.

    :param scale: Quantizer scale.
    :param zero_point: Quantizer zero point.
    :param tensor_type: Signed or Unsigned tensor type.
    :param axis: Axis for per-channel quantization. Should be none in case of per-tensor.
    """
    scale: np.ndarray
    zero_point: np.ndarray
    tensor_type: np.dtype
    axis: Optional[int] = None


def convert_fq_params_to_onnx_params(parameters: FakeQuantizeParameters,
                                     num_bits: int,
                                     mode: QuantizationMode,
                                     tensor_type: np.dtype,
                                     axis: Optional[int] = None) -> ONNXQuantizerLayerParameters:
    """
    Converts common FakeQuantizeParameters to ONNXQuantizerLayerParameters.

    :param parameters: FakeQuantizeParameters representation.
    :param num_bits: Number of quantizer bits.
    :param mode: Symmetric or Asymmetric mode.
    :param tensor_type: Value type of the tensor. Could be INT8 or UINT8.
    :param axis: Axis for per-channel quantization. Should be none in case of per-tensor.
    :return: Quantizer layer attributes.
    """
    input_low, input_high = parameters.input_low, parameters.input_high
    output_low, output_high = parameters.output_low, parameters.output_high
    if not np.allclose(input_high, output_high) or not np.allclose(input_low, output_low):
        raise ValueError('ONNX Quantize/Dequantize pairs only support'
                         ' input_high == output_high and input_low == output_low.')

    level_low, level_high = get_level_low_level_high(tensor_type, num_bits)
    levels = level_high - level_low + 1
    if levels not in [255, 256]:
        raise ValueError('Can only export to INT8/UIN8 256-level ONNX Quantize/Dequantize pairs.')

    scale, zero_point = calculate_scale_zero_point(input_low, input_high, level_low, level_high, mode)
    return ONNXQuantizerLayerParameters(scale, zero_point, tensor_type, axis)


def get_level_low_level_high(tensor_type: np.dtype, num_bits: int) -> Tuple[int, int]:
    """
    Returns the minimum and maximum level for the quantizer.
    In ONNX opset Q/DequantizeLinear-13 uses only two levels: [-128, 127] and [0, 255].

    :param tensor_type: Value type of the tensor. Could be INT8 or UINT8.
    :param num_bits: Number of quantizer bits.
    :return: Minimum level and maximum level of the quantizer.
    """
    if tensor_type == np.uint8:
        return 0, 2 ** num_bits - 1
    return - (2 ** num_bits) // 2, (2 ** num_bits) // 2 - 1


def calculate_scale_zero_point(input_low: np.ndarray, input_high: np.ndarray, level_low: int, level_high: int,
                               mode: QuantizationMode, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Quantizer/Dequantizer layer scale level.
    Returns scale and zero_point values for the quantizer.

    :param input_low: The minimum limit for an input value based on collected statistics.
    :param input_high: The maximum limit for an input value based on collected statistics.
    :param level_low: The minimum level in the integer range to quantize.
        The default is "0" for an unsigned range, and "-2^(bit-1)" for a signed one .
    :param level_high: The maximum level in the integer range to quantize.
        The default is "2^bits-1" for an unsigned range, and "2^(bit-1)-1" for a signed one.
    :param mode: Symmetric or Asymmetric mode.
    :param eps: The correction value for scale to avoid division by zero.
    :return: Scale and Zero point values.
    """
    scale, zero_point = None, None
    if mode == QuantizationMode.SYMMETRIC:
        scale = np.array((input_high - input_low) / (level_high - level_low))
        zero_point = np.zeros_like(scale, dtype=np.int32)
    elif mode == QuantizationMode.ASYMMETRIC:
        scale = np.array((input_high - input_low) / (level_high - level_low))
        zero_point = np.round(level_low - input_low / np.maximum(scale, eps)).astype(np.int32)

        level_low *= np.ones_like(zero_point, dtype=np.int32)
        level_high *= np.ones_like(zero_point, dtype=np.int32)

        zero_point = np.maximum(zero_point, level_low)
        zero_point = np.minimum(zero_point, level_high)

    scale = np.squeeze(scale).astype(np.float32)
    zero_point = np.squeeze(zero_point).astype(np.int32)

    return scale, zero_point
