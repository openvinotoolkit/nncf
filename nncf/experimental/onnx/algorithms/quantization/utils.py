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

from typing import Union
from typing import List

import onnx
import numpy as np

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxTensorStatistic


class QuantizerLayerParameters:
    """
    Class handles Quantizer/Dequantizer layer attributes.
    """

    def __init__(self, scale: List[float], zero_point: List[int], mode: QuantizationMode):
        self.scale = scale
        self.zero_point = zero_point
        self.mode = mode


def calculate_scale_level(max_val: Union[float, np.ndarray], min_val: Union[float, np.ndarray],
                          num_bits: int,
                          mode: QuantizationMode) -> Union[float, np.ndarray]:
    """
    Calculates Quantizer/Dequantizer layer scale level.
    """
    if mode == QuantizationMode.SYMMETRIC:
        input_abs_max = np.maximum(np.abs(max_val), np.abs(min_val))
        return input_abs_max / ((2 ** num_bits - 1) / 2)
    return (max_val - min_val) / 2 ** num_bits


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig) -> \
        QuantizerLayerParameters:
    """
    Calculates Quantizer/Dequantizer layer attributes for weight quantizer such as scale, zero_points and
    quantization mode: symmetric, asymmetric.
    """
    per_channel = quantizer_config.per_channel
    num_bits = quantizer_config.num_bits
    mode = quantizer_config.mode

    if per_channel:
        axes = tuple(range(len(weight_tensor.shape))[1:])
    else:
        axes = None
    input_high = np.amax(weight_tensor, axis=axes)
    input_low = np.amin(weight_tensor, axis=axes)
    scales = calculate_scale_level(input_high, input_low, num_bits, mode)
    zero_points = np.zeros_like(scales, dtype=np.int64)
    return QuantizerLayerParameters(scales.tolist(), zero_points.tolist(), mode)


def calculate_activation_quantizer_parameters(statistics: ONNXMinMaxTensorStatistic,
                                              quantizer_config: QuantizerConfig) -> QuantizerLayerParameters:
    """
    Calculates Quantizer/Dequantizer layer attributes for activation quantizer such as scale, zero_points and
    quantization mode: symmetric, asymmetric.
    """
    num_bits = quantizer_config.num_bits
    input_low = statistics.min_values
    input_high = statistics.max_values
    if input_low < 0:
        mode = QuantizationMode.SYMMETRIC
    else:
        mode = QuantizationMode.ASYMMETRIC

    scales = calculate_scale_level(input_high, input_low, num_bits, mode)
    zero_points = np.zeros_like(scales, dtype=np.int64)
    return QuantizerLayerParameters(scales.tolist(), zero_points.tolist(), mode)


def find_ignored_scopes(disallowed_op_types: List[str], model: onnx.ModelProto) -> List[str]:
    """
    Find ignored_scopes from disallowed_op_types.
    For example, if disallowed_op_types = {"Mul", "Add"},
    all nodes which have op_type = "Mul" or "Add" will be added to ignored_scopes.
    """
    disallowed_op_types = set(disallowed_op_types)
    return [node.name for node in model.graph.node if node.op_type in disallowed_op_types]
