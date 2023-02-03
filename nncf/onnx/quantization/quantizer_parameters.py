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


def calculate_scale_zero_point(params: QuantizerLayerParameters,
                               quantizer_config: QuantizerConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates Quantizer/Dequantizer layer scale level.
    Returns scale and zero_point values for the quantizer.

    :param params: Common representation of quantization parameters.
    :param quantizer_config: Config of the quantization configuration.
    :return: Scale and Zero point values.
    """
    if quantizer_config.signedness_to_force is not None:
        tensor_type = np.int8 if quantizer_config.signedness_to_force else np.uint8
    else:
        tensor_type = np.uint8 if np.all(params.input_low >= 0) else np.int8

    num_bits = quantizer_config.num_bits
    input_low = params.input_low
    input_high = params.input_high
    level_low, level_high = get_level_low_level_high(tensor_type, num_bits)

    scales = (input_high - input_low) / (level_high - level_low)
    zero_point = (level_low * input_high - level_high * input_low) / (input_high - input_low)

    level_low *= np.ones_like(zero_point, dtype=np.int32)
    level_high *= np.ones_like(zero_point, dtype=np.int32)

    zero_point = np.maximum(zero_point, level_low)
    zero_point = np.minimum(zero_point, level_high)

    scales = np.squeeze(scales).astype(tensor_type)
    zero_point = np.squeeze(zero_point).astype(tensor_type)
    return scales, zero_point
