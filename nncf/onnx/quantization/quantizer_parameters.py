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
    tensor_type: np.dtype
    axis: Optional[int] = None


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
                               mode: QuantizationMode, tensor_type: np.dtype,
                               eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Quantizer/Dequantizer layer scale level.
    Returns scale and zero_point values for the quantizer.

    :param input_low: Minimum limit for input value.
    :param input_high: Maximum limit for input value.
    :param level_low: The minimum level of quantizer quants.
    :param level_high: The maximum level of quantizer quants.
    :param mode: Symmetric or Asymmetric mode.
    :param tensor_type: Value type of the tensor. Could be INT8 or UINT8.
    :param eps: The correction value is added to the input range to avoid division by zero.
    :return: Scale and Zero point values.
    """
    input_range_safe = abs(level_high - level_low) + eps
    if mode == QuantizationMode.SYMMETRIC:
        input_low = np.zeros_like(input_high) if tensor_type == np.uint8 else -input_high
        scales = np.array((input_high - input_low) / (level_high - level_low))
        zero_point = np.zeros_like(scales, dtype=np.int32)
    else:
        scales = np.array((input_high - input_low) / input_range_safe)
        zero_point = np.round(level_low - input_low / scales).astype(np.int32)

        level_low *= np.ones_like(zero_point, dtype=np.int32)
        level_high *= np.ones_like(zero_point, dtype=np.int32)

        zero_point = np.maximum(zero_point, level_low)
        zero_point = np.minimum(zero_point, level_high)

    scales = np.squeeze(scales).astype(np.float32)
    zero_point = np.squeeze(zero_point).astype(np.int32)

    return scales, zero_point
