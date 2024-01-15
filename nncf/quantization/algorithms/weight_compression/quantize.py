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
from typing import Tuple

import numpy as np

from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.compression_info import WeightCompressionConfig
from nncf.quantization.fake_quantize import calculate_scale_zero_point


def do_integer_quantization(
    weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The method quantizes the given weights to integer data type in accordance with the compression config.
    The config defines a quantization mode:
        INT8_SYM mode refers to unsigned int8 symmetric weight compression with a fixed zero point equals to 128 -
            quantization to [0, 255] range.
        INT8_ASYM mode refers to unsigned int8 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 255] range.
        INT4_ASYM mode refers to unsigned int4 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 15] range.
        INT4_SYM mode refers to unsigned int4 symmetric weight compression with a fixed zero point equals to 8 -
            quantization to [0, 15] range.
        NF4 mode requires a dedicated procedure and it is not supported in this method.
    One of the parameter of compression config is a group size. Quantization is per-channel, if group size equals to -1,
    otherwise it's per-group, i.e. group size number of weights in the channel dimension share quantization parameters
    (scales).

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The compressed weights, scale and zero point that was used for its quantization.
    """
    mode = config.mode
    assert mode != CompressWeightsMode.NF4, "The function supports integer quantization only"
    group_size = config.group_size
    num_bits = config.num_bits

    level_low = 0
    level_high = 2**num_bits - 1

    if group_size != -1:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axis = reshape_weights_for_grouped_quantization(weight, reduction_axis, group_size)

    if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]:
        min_values = np.min(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = np.max(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
    else:
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
        level_low_sym = -(2 ** (num_bits - 1))
        level_high_sym = 2 ** (num_bits - 1) - 1
        scale = scale / level_high_sym
        zero_point = np.array([-level_low_sym])

    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    compressed_weights = np.round(weight / scale + zero_point)
    compressed_weights = np.clip(compressed_weights, level_low, level_high).astype(np.uint8)
    return compressed_weights, scale, zero_point


def get_integer_quantization_error(weight: np.ndarray, reduction_axis: int, config: WeightCompressionConfig) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    orig_shape = weight.shape
    compressed_weights, scale, zero_point = do_integer_quantization(weight, reduction_axis, config)

    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (compressed_weights - zero_point) * scale

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = np.mean(diff, axis=reduction_axis)
    val = np.max(layer_err)
    return val


def reshape_weights_for_grouped_quantization(
    weight: np.ndarray, reduction_axis: int, group_size: int
) -> Tuple[np.ndarray, int]:
    """
    Reshapes weights for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weights with shapes [c_out, c_in] and group size = 128, shape of reshaped weights is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weights and new reduction axis.
    """
    assert group_size != -1
    assert isinstance(reduction_axis, int)
    channel_size = weight.shape[reduction_axis]
    if channel_size % group_size != 0:
        raise RuntimeError(f"Channel size {channel_size} should be divisible by size of group {group_size}")

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axis : reduction_axis + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axis += 1
    return reshaped_weight, reduction_axis


def get_norm_weight_and_nf4_scale(
    weight: np.ndarray, reduction_axis: int, group_size: int = -1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates scale for nf4 quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :return: Normalized weights and nf4 scale.
    """
    if group_size != -1:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axis = reshape_weights_for_grouped_quantization(weight, reduction_axis, group_size)
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    else:
        scale = np.max(np.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, 1, a2]
    eps = np.finfo(weight.dtype).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale[np.abs(scale) < eps] = eps
    norm_weight = weight / scale
    return norm_weight, scale
