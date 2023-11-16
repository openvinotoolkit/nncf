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
from typing import Optional, Tuple

import numpy as np

from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.fake_quantize import calculate_scale_zero_point


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


def convert_fq_params_to_onnx_params(
    parameters: FakeQuantizeParameters, num_bits: int, tensor_type: np.dtype, axis: Optional[int] = None
) -> ONNXQuantizerLayerParameters:
    """
    Converts common FakeQuantizeParameters to ONNXQuantizerLayerParameters.

    :param parameters: FakeQuantizeParameters representation.
    :param num_bits: Number of quantizer bits.
    :param tensor_type: Value type of the tensor. Could be INT8 or UINT8.
    :param axis: Axis for per-channel quantization. Should be none in case of per-tensor.
    :return: Quantizer layer attributes.
    """
    if num_bits != 8:
        raise ValueError("Can only export to INT8/UIN8 8 bits ONNX Quantize/Dequantize pairs.")

    levels = parameters.levels
    if levels not in [255, 256]:
        raise ValueError("Can only export to INT8/UIN8 256-level ONNX Quantize/Dequantize pairs.")

    input_low, input_high = parameters.input_low.data, parameters.input_high.data
    output_low, output_high = parameters.output_low.data, parameters.output_high.data
    if not np.allclose(input_high, output_high) or not np.allclose(input_low, output_low):
        raise ValueError(
            "ONNX Quantize/Dequantize pairs only support input_high == output_high and input_low == output_low."
        )

    level_low, level_high = get_level_low_level_high(tensor_type)
    narrow_range = levels == 2**num_bits - 1
    scale, zero_point = calculate_scale_zero_point(input_low, input_high, level_low, level_high, narrow_range)
    return ONNXQuantizerLayerParameters(scale, zero_point, tensor_type, axis)


def get_level_low_level_high(tensor_type: np.dtype) -> Tuple[int, int]:
    """
    Returns the minimum and maximum level for the quantizer.
    In ONNX opset Q/DequantizeLinear-13 uses only two levels: [-128, 127] and [0, 255].

    :param tensor_type: Value type of the tensor. Could be INT8 or UINT8.
    :return: Minimum level and maximum level of the quantizer.
    """
    return (0, 255) if tensor_type == np.uint8 else (-128, 127)
