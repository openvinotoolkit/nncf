from typing import Union
from typing import List
from typing import Tuple

import numpy as np

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode

from nncf.experimental.post_training.statistics.statistics_collector import MinMaxLayerStatistic


class QuantizerLayerParameters:
    def __init__(self, scale: List[float], zero_point: List[float], mode: QuantizationMode):
        self.scale = scale
        self.zero_point = zero_point
        self.mode = mode


def calculate_scale_level(
        max_val: Union[float, np.ndarray],
        min_val: Union[float, np.ndarray],
        num_bits: int,
        mode: QuantizationMode):
    # Always full range
    if mode == QuantizationMode.SYMMETRIC:
        input_abs_max = np.maximum(np.abs(max_val), np.abs(min_val))
        return input_abs_max / ((2 ** num_bits - 1) / 2)
    return (max_val - min_val) / 2 ** num_bits


def calculate_weight_quantizer_parameters(weight_tensor: np.ndarray, quantizer_config: QuantizerConfig) -> \
        QuantizerLayerParameters:
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
    zero_points = np.zeros_like(scales, dtype=np.int)
    return QuantizerLayerParameters(scales.tolist(), zero_points.tolist(), mode)


def calculate_activation_quantizer_parameters(layer_statistics: MinMaxLayerStatistic,
                                              quantizer_config: QuantizerConfig) -> \
        QuantizerLayerParameters:
    # TODO:PERCHANNEL IS NOT SUPPORTED.
    per_channel = quantizer_config.per_channel
    num_bits = quantizer_config.num_bits
    mode = quantizer_config.mode

    # if per_channel:
    #     axes = tuple(range(len(weight_tensor.shape))[1:])
    # else:
    #     axes = None

    axes = None
    input_high = layer_statistics.get_global_max_value()
    input_low = layer_statistics.get_global_min_value()
    if input_low < 0:
        mode = QuantizationMode.SYMMETRIC
    else:
        mode = QuantizationMode.ASYMMETRIC

    scales = calculate_scale_level(input_high, input_low, num_bits, mode)
    zero_points = np.zeros_like(scales, dtype=np.int)
    return QuantizerLayerParameters(scales.tolist(), zero_points.tolist(), mode)
