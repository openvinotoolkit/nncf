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

import numpy as np

import nncf
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorDataType

ReductionAxes = Tuple[int, ...]

NF4_QUANTILES = np.array(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=np.float32,
)
CENTER_OF_NF4_QUANTILES = np.array(
    [
        -0.8480964004993439,
        -0.6106329262256622,
        -0.4599952697753906,
        -0.33967943489551544,
        -0.23460740596055984,
        -0.13791173323988914,
        -0.045525018125772476,
        0.03979014977812767,
        0.1202552504837513,
        0.2035212516784668,
        0.2920137718319893,
        0.3893125355243683,
        0.5016634166240692,
        0.6427869200706482,
        0.8614784181118011,
    ],
    dtype=np.float32,
)


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


def reshape_weight_for_grouped_quantization(
    weight: Tensor, reduction_axes: ReductionAxes, group_size: int
) -> Tuple[Tensor, int]:
    """
    Reshapes weight for group-wise quantization and return a new reduction axis for collecting statistics per group
    dimension. Having weight with shapes [c_out, c_in] and group size = 128, shape of reshaped weight is
    [c_out, c_in // 128, 128].

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weight and new reduction axis.
    """
    assert group_size != -1
    if isinstance(reduction_axes, tuple) and len(reduction_axes) == 1:
        reduction_axes = reduction_axes[0]
    if not isinstance(reduction_axes, int):
        raise NotImplementedError(
            f"Group-wise quantization expects a single reduction axis, but given: {reduction_axes}."
        )
    channel_size = weight.shape[reduction_axes]
    if channel_size % group_size != 0:
        raise nncf.ValidationError(f"Channel size {channel_size} should be divisible by size of group {group_size}")

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axes : reduction_axes + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axes += 1
    return reshaped_weight, reduction_axes


def calculate_nf4_scale(weight: Tensor, reduction_axes: ReductionAxes) -> Tensor:
    """
    Calculates the scale for nf4 quantization.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes along which to reduce (collect) different statistics (e.g., min, max).
    :return: Scale tensor of float32 type for nf4 quantization.
    """
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    scale = fns.max(fns.abs(weight), axis=reduction_axes, keepdims=True)

    # NOTE: adding machine epsilon to avoid division by zero
    eps = fns.finfo(weight).eps
    scale = fns.where(fns.abs(scale) < eps, eps, scale)

    return scale


def calculate_e2m1_scale(weight: Tensor, reduction_axes: ReductionAxes, max_val=6.0) -> Tensor:
    """
    Calculates the scale for e2m1 quantization.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes along which to reduce (collect) different statistics (e.g., min, max).
    :param max_val: Maximal value of e2m1 type.
    :param to_e8m0: Defines convert scale to e8m0 or not.
    :return: Scale tensor of float32 type for e2m1 quantization.
    """
    scale = calculate_nf4_scale(weight, reduction_axes) / max_val

    scale = fns.log2(scale)
    scale = fns.ceil(scale)
    scale = fns.clip(scale, -127, 127)
    scale = 2**scale

    return scale


def calculate_normalized_weight(weight: Tensor, scale: Tensor) -> Tensor:
    """
    Normalizes the weight tensor using the provided scale.

    :param weight: Weight tensor to normalize.
    :param scale: Scale tensor used for normalization.
    :return: Normalized weight tensor.
    """
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)
    if scale.dtype != TensorDataType.float32:
        scale = scale.astype(TensorDataType.float32)

    return weight / scale


def calculate_nf4_weight(weight: Tensor, scale: Tensor) -> Tensor:
    """
    Quantizes the weight tensor to NF4 format.

    :param weight: Weight tensor to quantize.
    :param scale: Scale tensor used for normalization.
    :return: Quantized weight tensor in NF4 format.
    """
    norm_weight = calculate_normalized_weight(weight, scale)

    center_nf4_quantiles = fns.from_numpy(CENTER_OF_NF4_QUANTILES, backend=norm_weight.backend)
    nf4_quantiles = fns.from_numpy(NF4_QUANTILES, backend=norm_weight.backend)

    index_of_quantile = fns.searchsorted(center_nf4_quantiles, norm_weight)
    nf4_weight = nf4_quantiles[index_of_quantile]

    return nf4_weight


def decompress_nf4_weight(weight: Tensor, scale: Tensor) -> Tensor:
    """
    Decompresses the NF4 quantized weight tensor.

    :param weight: Quantized weight tensor in NF4 format.
    :param scale: Scale tensor used for decompression.
    :return: Decompressed weight tensor.
    """
    return weight * scale


def calculate_normalized_weight_and_fp4_scale(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    group_size: int = -1,
    precomputed_scale: Tensor = None,
    mode: CompressWeightsMode = CompressWeightsMode.NF4,
) -> Tuple[Tensor, Tensor]:
    """
    Calculates scale for fp4 (nf4, e2m1) quantization and normalizes weights by the scale.
    Weights are reshaped in case of positive value of group size.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping. Defaults to -1.
    :param precomputed_scale: Precomputed scale.
    :return: Normalized weight tensor of float32 type and nf4 scale tensor of float32 type.
    """
    assert mode in [CompressWeightsMode.NF4, CompressWeightsMode.E2M1]
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if group_size != -1:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, group_size)

    if mode == CompressWeightsMode.NF4:
        scale = calculate_nf4_scale(weight, reduction_axes) if precomputed_scale is None else precomputed_scale
    if mode == CompressWeightsMode.E2M1:
        scale = calculate_e2m1_scale(weight, reduction_axes) if precomputed_scale is None else precomputed_scale
    norm_weight = calculate_normalized_weight(weight, scale)
    return norm_weight, scale


def calculate_integer_quantization_params(
    weight: Tensor, reduction_axes: ReductionAxes, config: WeightCompressionConfig
) -> Tuple[Tensor, Tensor]:
    """
    Calculates the scale and zero point for integer quantization.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Weight compression configuration.
    :return: Scale and zero point tensors.
    """
    mode = config.mode
    assert config.is_integer(), "The function supports integer quantization only"
    num_bits = config.num_bits

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]:
        level_low = 0
        level_high = 2**num_bits - 1
        min_values = fns.min(weight, axis=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = fns.max(weight, axis=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
        return scale, zero_point

    level_high = 2 ** (num_bits - 1) - 1
    scale = fns.max(fns.abs(weight), axis=reduction_axes, keepdims=True)  # [a1, r//gs, 1, a2]
    scale /= level_high
    eps = fns.finfo(scale).eps
    # NOTE: adding machine epsilon to avoid division by zero
    scale = fns.where(fns.abs(scale) < eps, eps, scale)
    return scale, None


def calculate_quantized_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    scale: Tensor,
    zero_point: Optional[Tensor] = None,
    invert_scale=False,
) -> Tensor:
    """
    Quantizes the weight tensor using the provided scale and zero point.

    :param weight: Weight tensor to quantize.
    :param config: Weight compression configuration.
    :param scale: Scale tensor used for quantization.
    :param zero_point: Zero point tensor used for quantization.
    :param invert_scale: applies inversion for scale and then multiply by weights instead of division.
    :return: Quantized weight tensor of uint8 or int8 type.
    """
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)
    if scale.dtype != TensorDataType.float32:
        scale = scale.astype(TensorDataType.float32)

    num_bits = config.num_bits
    asym_quant = config.mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]
    dtype = TensorDataType.uint8 if asym_quant else TensorDataType.int8
    level_low = 0 if asym_quant else -(2 ** (num_bits - 1))
    level_high = 2**num_bits - 1 if asym_quant else 2 ** (num_bits - 1) - 1

    if invert_scale:
        scale = fns.power(scale, -1)
        compressed_weights = weight * scale
    else:
        compressed_weights = weight / scale
    if zero_point is not None:
        compressed_weights += zero_point.astype(weight.dtype)
    compressed_weights = fns.round(compressed_weights)
    compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(dtype)
    return compressed_weights


def do_integer_quantization(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
    invert_scale=False,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    The method quantizes the given weights to integer data type in accordance with the compression config.
    The config defines a quantization mode:
        INT8_SYM mode refers to signed int8 symmetric weight compression without zero point -
            quantization to [-128, 127] range.
        INT8_ASYM mode refers to unsigned int8 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 255] range.
        INT4_ASYM mode refers to unsigned int4 asymmetric weight compression with a typical non-fixed zero-point -
            quantization to [0, 15] range.
        INT4_SYM mode refers to signed int4 symmetric weight compression without zero point -
            quantization to [-8, 7] range.
        NF4 or E2M1 mode requires a dedicated procedure and it is not supported in this method.
    One of the parameter of compression config is a group size. Quantization is per-channel, if group size equals to -1,
    otherwise it's per-group, i.e. group size number of weights in the channel dimension share quantization parameters
    (scales).

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :param precomputed_scale: Precomputed scale.
    :param precomputed_zero_point: Precomputed zero point.
    :param invert_scale: applies inversion for scale and then multiply by weights instead of division.
        Need as reference implementation for OV.
    :return: The compressed weights tensor of uint8 (asymmetric mode) or int8 (symmetric mode) type,
        scale tensor of float32 type and zero point tensor of int32 type that was used for its quantization.
    """
    assert config.is_integer(), "The function supports integer quantization only"
    group_size = config.group_size

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if group_size != -1:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, group_size)

    if precomputed_zero_point is None or precomputed_zero_point is None:
        scale, zero_point = calculate_integer_quantization_params(weight, reduction_axes, config)
    if precomputed_scale is not None:
        scale = precomputed_scale
    if precomputed_zero_point is not None:
        zero_point = precomputed_zero_point

    compressed_weights = calculate_quantized_weight(weight, config, scale, zero_point, invert_scale)
    return compressed_weights, scale, zero_point


def get_integer_quantization_error(
    weight: Tensor, reduction_axes: ReductionAxes, config: WeightCompressionConfig
) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    orig_shape = weight.shape

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    compressed_weights, scale, zero_point = do_integer_quantization(weight, reduction_axes, config)
    decompressed_weight = do_dequantization(compressed_weights, scale, zero_point)

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = fns.mean(diff, axis=reduction_axes)
    val = fns.max(layer_err)
    return val.item()


def compress_weight(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
):
    """
    Compress weight using compression configuration.

    :param weight: The weight to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Compression configuration.
    :param precomputed_scale: Precomputed scale.
    :param precomputed_zero_point: Precomputed zero point.
    :return: The compressed weight and decompression parameters as instance of CompressedWeight
    """
    if not config.is_integer():
        compressed_weight, scale = calculate_normalized_weight_and_fp4_scale(
            weight, reduction_axes, config.group_size, precomputed_scale, config.mode
        )
        return CompressedWeight(compressed_weight, scale)
    compressed_weight, scale, zero_point = do_integer_quantization(
        weight, reduction_axes, config, precomputed_scale, precomputed_zero_point
    )

    return CompressedWeight(compressed_weight, scale, zero_point)


def do_dequantization(
    compressed_weights: Tensor, scale: Tensor, zero_point: Optional[Tensor] = None, reduction_axis: int = -1
) -> Tensor:
    """
    The method dequantizes the given weights to float point data type in accordance with the scale and
    zero_point data type.

    :param compressed_weights: compressed weights.
    :param scale: scale in compression/quantization.
    :param zero_point: zero point in compression/quantization.
    :param reduction_axis: axis for return back for group compression.
    :return: dequantized/decompressed weights.
    """
    decompressed_weight = compressed_weights - zero_point if zero_point is not None else compressed_weights
    decompressed_weight = decompressed_weight.astype(scale.dtype) * scale

    if reduction_axis > -1:
        shape = list(decompressed_weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
        shape[reduction_axis] = shape[reduction_axis] * shape[reduction_axis + 1]
        shape[reduction_axis + 1] = 1
        reshaped_weight = decompressed_weight.reshape(shape)
        reshaped_weight = fns.squeeze(reshaped_weight)
        decompressed_weight = reshaped_weight

    return decompressed_weight
