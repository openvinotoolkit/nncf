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
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic


@dataclass
class ONNXQuantizerLayerParameters:
    """
    Class handles Quantizer layer attributes.

    :param scale: Quantizer scale.
    :param zero_point: Quantizer zero point.
    :param mode: Quantizer mode. Could be Symmetric or Asymmetric.
    :param axis: Axis for per-channel quantization. Should be none in case of per-tensor.
    :param tensor_type: Signed or Unsigned tensor type.
    """
    scale: np.ndarray
    zero_point: np.ndarray
    mode: QuantizationMode
    axis: Optional[int] = None
    tensor_type: Optional[np.dtype] = None


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


def calculate_scale_zero_point(max_val: np.ndarray, min_val: np.ndarray, level_low: int, level_high: int,
                               mode: QuantizationMode, tensor_type: np.dtype) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Quantizer/Dequantizer layer scale level.
    Returns scale and zero_point values foe the quantizer.

    :param max_val: The maximum value of the input tensor.
    :param min_val: The minimum value of the input tensor.
    :param level_low: The minimum level of quantizer quants.
    :param level_high: The maximum level of quantizer quants.
    :param mode: Symmetric or Asymmetric mode.
    :param tensor_type: Value type of the tensor. Could be INT8 or UINT8.
    :return: Scale and Zero point values.
    """
    scale, zero_point = None, None
    if mode == QuantizationMode.SYMMETRIC:
        input_abs_max = np.maximum(np.abs(max_val), np.abs(min_val))
        max_val = input_abs_max
        min_val = np.zeros_like(input_abs_max) if tensor_type == np.uint8 else -input_abs_max
        scale = np.array((max_val - min_val) / float(level_high - level_low))
        zero_point = np.zeros_like(scale, dtype=np.int32)
    elif mode == QuantizationMode.ASYMMETRIC:
        scale = np.array((max_val - min_val) / float(level_high - level_low))
        zero_point = np.round(level_low - min_val / scale).astype(np.int32)

        level_low *= np.ones_like(zero_point, dtype=np.int32)
        level_high *= np.ones_like(zero_point, dtype=np.int32)

        zero_point = np.maximum(zero_point, level_low)
        zero_point = np.minimum(zero_point, level_high)

    return scale, zero_point


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig,
                                          axis: Optional[int]) -> ONNXQuantizerLayerParameters:
    """
    Calculates Quantizer/Dequantizer layer attributes for weight quantizer such as scale, zero_points and
    quantization mode: symmetric, asymmetric.

    :param weight_tensor: Weight tensor to calculate quantizer attributes.
    :param quantizer_config: Config of Quantizer.
    :param axis: In per-channel case - the axis for the quantization. In per-tensor - ignored.
    :return: Parameters of Quantizer.
    """
    if quantizer_config.per_channel:
        assert axis is not None
        axes = list(range(len(weight_tensor.shape)))
        axes.pop(axis)
        axes = tuple(axes)
    else:
        axes = None
    # The weight is restricted to have only signed range.
    tensor_type = np.int8
    if quantizer_config.signedness_to_force is not None and not quantizer_config.signedness_to_force:
        raise ValueError('The HW expects to have signed quantization of weights, '
                         'while the quantizer configuration for weights contains signedness_to_force=False.')
    input_high = np.amax(weight_tensor, axis=axes)
    input_low = np.amin(weight_tensor, axis=axes)
    level_low, level_high = get_level_low_level_high(tensor_type, quantizer_config.num_bits)
    scales, zero_points = calculate_scale_zero_point(input_high, input_low, level_low, level_high,
                                                     quantizer_config.mode, tensor_type)
    return ONNXQuantizerLayerParameters(scales, zero_points, quantizer_config.mode, axis, tensor_type)


def calculate_activation_quantizer_parameters(statistics: MinMaxTensorStatistic,
                                              quantizer_config: QuantizerConfig,
                                              axis: Optional[int] = None) -> ONNXQuantizerLayerParameters:
    """
    Calculates Quantizer/Dequantizer layer attributes for activation quantizer such as scale, zero_points and
    quantization mode: symmetric, asymmetric.

    :param statistics: Collected statistics for the quantized insertion.
    :param quantizer_config: Config of the quantization configuration.
    :param axis: Axis of the quantization. None in a per-tensor quantization case.
    :return: Parameters of the quantizer/dequantizer layers.
    """
    input_low = np.array(statistics.min_values)
    input_high = np.array(statistics.max_values)
    tensor_type = np.uint8 if np.all(input_low >= 0) else np.int8
    if quantizer_config.signedness_to_force is not None:
        tensor_type = np.int8 if quantizer_config.signedness_to_force else np.uint8
    level_low, level_high = get_level_low_level_high(tensor_type, quantizer_config.num_bits)
    scales, zero_points = calculate_scale_zero_point(input_high, input_low, level_low, level_high,
                                                     quantizer_config.mode, tensor_type)
    return ONNXQuantizerLayerParameters(scales, zero_points, quantizer_config.mode, axis, tensor_type)
