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

from dataclasses import dataclass
from typing import Optional, Tuple

import nncf
from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor.definitions import TensorDataType
from nncf.experimental.tensor.functions import numeric as fns
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.fake_quantize import calculate_scale_zero_point


@dataclass
class CompressedWeight:
    """
    Compressed weight and decompression parameters.

    :param tensor: The tensor with compressed weight.
    :param scale: The decompression scale, in practice it is dequantization scale for the INT quantization.
    :param zero_point: The zero-point, it is the value of the compression type corresponding to the value 0
        in the non-compression realm. Applicable for INT quantization.
    """

    tensor: Tensor
    scale: Tensor
    zero_point: Optional[Tensor] = None


def reshape_weight_for_grouped_quantization(weight: Tensor, reduction_axis: int, group_size: int) -> Tuple[Tensor, int]:
    """
    Reshapes weight for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weight with shapes [c_out, c_in] and group size = 128, shape of reshaped weight is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weight and new reduction axis.
    """
    assert group_size != -1
    assert isinstance(reduction_axis, int)
    channel_size = weight.shape[reduction_axis]
    if channel_size % group_size != 0:
        raise nncf.ValidationError(f"Channel size {channel_size} should be divisible by size of group {group_size}")

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axis : reduction_axis + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axis += 1
    return reshaped_weight, reduction_axis


def calculate_normalized_weight_and_nf4_scale(
    weight: Tensor, reduction_axis: int, group_size: int = -1
) -> Tuple[Tensor, Tensor]:
    """
    Calculates scale for nf4 quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :return: Normalized weight tensor of float32 type and nf4 scale tensor of float32 type.
    """
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if group_size != -1:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axis = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)
        scale = fns.max(fns.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
    else:
        scale = fns.max(fns.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, 1, a2]
    eps = fns.finfo(weight).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale = fns.where(fns.abs(scale) < eps, eps, scale)
    norm_weight = weight / scale
    return norm_weight, scale


def do_integer_quantization(
    weight: Tensor, reduction_axis: int, config: WeightCompressionConfig
) -> Tuple[Tensor, Tensor, Tensor]:
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
    :return: The compressed weights tensor of uint8 type, scale tensor of float32 type and
        zero point tensor of int32 type that was used for its quantization.
    """
    mode = config.mode
    assert mode != CompressWeightsMode.NF4, "The function supports integer quantization only"
    group_size = config.group_size
    num_bits = config.num_bits

    level_low = 0
    level_high = 2**num_bits - 1

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if group_size != -1:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axis = reshape_weight_for_grouped_quantization(weight, reduction_axis, group_size)

    if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]:
        min_values = fns.min(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = fns.max(weight, axis=reduction_axis, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
    else:
        scale = fns.max(fns.abs(weight), axis=reduction_axis, keepdims=True)  # [a1, r//gs, 1, a2]
        level_low_sym = -(2 ** (num_bits - 1))
        level_high_sym = 2 ** (num_bits - 1) - 1
        scale = scale / level_high_sym
        zero_point = fns.as_tensor_like(scale, [-level_low_sym])
        eps = fns.finfo(scale).eps
        # NOTE: adding machine epsilon to avoid division by zero
        scale = fns.where(fns.abs(scale) < eps, eps, scale)

    compressed_weights = fns.round(weight / scale + zero_point.astype(weight.dtype))
    compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(TensorDataType.uint8)
    return compressed_weights, scale, zero_point


def get_integer_quantization_error(weight: Tensor, reduction_axis: int, config: WeightCompressionConfig) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    :param weight: Weight array to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    orig_shape = weight.shape

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    compressed_weights, scale, zero_point = do_integer_quantization(weight, reduction_axis, config)

    decompressed_weight = (compressed_weights - zero_point).astype(weight.dtype) * scale

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = fns.mean(diff, axis=reduction_axis)
    val = fns.max(layer_err)
    return val.item()


def compress_weight(weight: Tensor, reduction_axis: int, config: WeightCompressionConfig):
    """
    Compress weight using compression configuration.

    :param weight: The weight to compress.
    :param reduction_axis: Axis, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Compresssion configuration.
    :return: The compressed weight and decompression parameters as instance of CompressedWeight
    """
    if config.mode == CompressWeightsMode.NF4:
        compressed_weight, scale = calculate_normalized_weight_and_nf4_scale(weight, reduction_axis, config.group_size)
        return CompressedWeight(compressed_weight, scale)

    compressed_weight, scale, zero_point = do_integer_quantization(weight, reduction_axis, config)
    return CompressedWeight(compressed_weight, scale, zero_point)


def do_dequantization(compressed_weights: Tensor, scale: Tensor, zero_point: Tensor) -> Tensor:
    """
    The method dequantizes the given weights to float point data type in accordance with the scale and
    zero_point data type.

    :param compressed_weights: compressed weights.
    :param scale: scale in compression/quantization.
    :param zero_point: zero point in compression/quantization.
    :return: dequantized/decompressed weights.
    """
    decompressed_weight = compressed_weights.astype(dtype=scale.dtype)
    decompressed_weight = (decompressed_weight - zero_point) * scale
    return decompressed_weight
