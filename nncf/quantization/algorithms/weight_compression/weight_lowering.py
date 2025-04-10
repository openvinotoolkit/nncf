# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

import nncf
from nncf.common.logging.logger import nncf_logger
from nncf.common.utils.backend import is_openvino_at_least
from nncf.common.utils.backend import is_openvino_available
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType

ReductionAxes = Union[int, Tuple[int, ...]]

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

CENTER_OF_NF4_QUANTILES = (NF4_QUANTILES[1:] + NF4_QUANTILES[:-1]) / 2


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
    Reshapes weight for group-wise quantization and return a reduction axis for collecting statistics per group
    dimension. Having a transposed weight with shapes [c_out, c_in] and group size = 128, shape of reshaped weight is
    [c_out, c_in // 128, 128], reduction axis = 1 and the returned reduction axis = 2.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param group_size: Number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
    :return: reshaped weight and new reduction axis.
    """
    assert group_size != -1
    if isinstance(reduction_axes, tuple) and len(reduction_axes) == 1:
        reduction_axes = reduction_axes[0]
    if not isinstance(reduction_axes, int):
        msg = f"Group-wise quantization expects a single reduction axis, but given: {reduction_axes}."
        raise nncf.UnsupportedModelError(msg)
    channel_size = weight.shape[reduction_axes]
    if channel_size % group_size != 0:
        msg = f"Channel size {channel_size} should be divisible by size of group {group_size}."
        raise nncf.InvalidGroupSizeError(msg)

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


def calculate_signed_scale(weight: Tensor, reduction_axes: ReductionAxes, num_bits=4) -> Tensor:
    """
    Calculates the signed scale for symmetric quantization.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes along which to reduce (collect) different statistics (e.g., min, max).
    :param num_bits: number of bits in compression.
    :return: Scale tensor.
    """
    factor = 2 ** (num_bits - 1)

    w_abs_min = fns.abs(fns.min(weight, axis=reduction_axes, keepdims=True))
    w_max = fns.max(weight, axis=reduction_axes, keepdims=True)

    scale = fns.where(w_abs_min >= w_max, w_abs_min, -w_max)
    scale /= factor

    eps = fns.finfo(scale).eps
    scale = fns.where(fns.abs(scale) < eps, eps, scale)

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


def do_nf4_quantization(weight: Tensor, scale: Tensor, is_normalized_weight: bool = False) -> Tensor:
    """
    Performs NF4 quantization. The floating point values are represented by floating point scale and look-up with
    16 floating-point values on [-1, 1]. Scale normalizes original values to [-1, 1] interval and look-up table
    "rounds" or "quantize" to the closest quant.

    :param weight: Weight tensor to quantize.
    :param scale: Scale tensor used for normalization.
    :param is_normalized_weight: Whether weight was scaled to [-1, 1] interval. Defaults to False.
    :return: Tensor with floating-point values, where each of them corresponds to 1 out of 16 quants on [-1, 1].
    """
    norm_weight = weight if is_normalized_weight else calculate_normalized_weight(weight, scale)
    center_nf4_quantiles = fns.from_numpy(CENTER_OF_NF4_QUANTILES, backend=norm_weight.backend)
    indexes = fns.searchsorted(center_nf4_quantiles, norm_weight)
    nf4_quantiles = fns.from_numpy(NF4_QUANTILES, backend=indexes.backend)
    nf4_weight = nf4_quantiles[indexes]
    return nf4_weight


def do_nf4_dequantization(nf4_weight: Tensor, scale: Tensor, reduction_axis: int = -1) -> Tensor:
    """
    Decompresses the NF4 quantized weight tensor.

    :param nf4_weight: Tensor with floating-point values,
        where each of them corresponds to 1 out of 16 quants on [-1, 1].
    :param scale: Scale tensor used for decompression.
    :param reduction_axis: axis along which weights were reshaped for group quantization and will be reshaped back to
        original shapes. If equals to -1, weights are not reshaped, assumed not a group quantization. Defaults to -1.
    :return: Decompressed weight tensor.
    """
    decompressed_weight = nf4_weight * scale
    if reduction_axis != -1:
        decompressed_weight = ungroup_weights(decompressed_weight, reduction_axis)
    return decompressed_weight


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
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
) -> Tuple[Tensor, Tensor]:
    """
    Calculates the scale and zero point for uniform quantization (INT4, INT8), when the range of values is divided into
    equal intervals, and each interval is assigned a quant.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Weight compression configuration.
    :return: Scale and zero point tensors.
    """
    if not config.is_integer:
        msg = "The function supports integer quantization only"
        raise nncf.InternalError(msg)

    num_bits = config.num_bits

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    if config.is_asym_mode:
        level_low = 0
        level_high = 2**num_bits - 1
        min_values = fns.min(weight, axis=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        max_values = fns.max(weight, axis=reduction_axes, keepdims=True)  # [a1, r, a2] -> [a1, 1, a2]
        scale, zero_point = calculate_scale_zero_point(
            min_values, max_values, level_low, level_high, narrow_range=False
        )
        return scale, zero_point

    scale = calculate_signed_scale(weight, reduction_axes, num_bits)
    return scale, None


def calculate_quantized_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    scale: Tensor,
    zero_point: Optional[Tensor] = None,
) -> Tensor:
    """
    Quantizes the weight tensor using the provided scale and zero point.

    :param weight: Weight tensor to quantize.
    :param config: Weight compression configuration.
    :param scale: Scale tensor used for quantization.
    :param zero_point: Zero point tensor used for quantization.
    :return: Quantized weight tensor of uint8 or int8 type.
    """
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)
    if scale.dtype != TensorDataType.float32:
        scale = scale.astype(TensorDataType.float32)

    num_bits = config.num_bits
    asym_quant = config.is_asym_mode
    dtype = TensorDataType.uint8 if asym_quant else TensorDataType.int8
    level_low = 0 if asym_quant else -(2 ** (num_bits - 1))
    level_high = 2**num_bits - 1 if asym_quant else 2 ** (num_bits - 1) - 1

    compressed_weights = weight / scale
    if zero_point is not None:
        compressed_weights += zero_point.astype(weight.dtype)
    compressed_weights = fns.round(compressed_weights)
    compressed_weights = fns.clip(compressed_weights, level_low, level_high).astype(dtype)

    return compressed_weights


def get_integer_quantization_error(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
) -> float:
    """
    Calculates a quantity characterizing the difference between floating point weights and fake quantized
    (compressed and decompressed) to integer ones.

    The error is computed as follows:
    error = max(mean((decompressed_weight - weight)^2, axis=reduction_axes))

    :param weight: Weight array to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Information on how to compress (quantize) a specific weight.
    :return: The quantity characterizing the error of integer quantization.
    """
    # Optimized implementation
    if _can_run_optimized(weight.backend):
        from nncf.openvino.optimized_functions import (
            get_integer_quantization_error as get_integer_quantization_error_ov,
        )

        return get_integer_quantization_error_ov(weight, reduction_axes, config)

    if weight.backend == TensorBackend.ov:
        weight = weight.as_numpy_tensor()
    orig_shape = weight.shape

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    decompressed_weight = quantize_dequantize_weight(weight, config, reduction_axes)

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
) -> CompressedWeight:
    """
    Compress weight using compression configuration.

    :param weight: The weight to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Compression configuration.
    :param precomputed_scale: Precomputed scale.
    :param precomputed_zero_point: Precomputed zero point.
    :return: The compressed weight and decompression parameters as instance of CompressedWeight
    """
    if not config.is_integer:
        if weight.backend == TensorBackend.ov:
            weight = weight.as_numpy_tensor()

        compressed_weight, scale = calculate_normalized_weight_and_fp4_scale(
            weight, reduction_axes, config.group_size, precomputed_scale, config.mode
        )
        return CompressedWeight(compressed_weight, scale)
    compressed_weight, scale, zero_point = do_int_quantization(
        weight, config, reduction_axes, precomputed_scale, precomputed_zero_point
    )

    return CompressedWeight(compressed_weight, scale, zero_point)


def ungroup_weights(weights: Tensor, reduction_axis: int) -> Tensor:
    """
    Reshapes weights used for group quantization back to original shape.

    :param weights: The weight to reshape.
    :param reduction_axis: The axis, along which weights were reshaped for group quantization and will be reshaped back
        to original shapes. If equals to -1, weights are not reshaped, assumed not a group quantization. Default to -1.
    :return: Reshaped weight.
    """
    shape = list(weights.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axis] = shape[reduction_axis] * shape[reduction_axis + 1]
    shape[reduction_axis + 1] = 1
    reshaped_weight = weights.reshape(shape)
    reshaped_weight = fns.squeeze(reshaped_weight)
    weights = reshaped_weight
    return weights


def do_int_dequantization(
    compressed_weights: Tensor, scale: Tensor, zero_point: Optional[Tensor] = None, reduction_axis: int = -1
) -> Tensor:
    """
    The method dequantizes the given weights to float point data type in accordance with the scale and
    zero_point data type.

    :param compressed_weights: compressed weights.
    :param scale: scale in compression/quantization.
    :param zero_point: zero point in compression/quantization.
    :param reduction_axis: axis along which weights were reshaped for group quantization and will be reshaped back to
        original shapes. If equals to -1: weights are not reshaped, assumed not a group quantization. Default to -1.
    :return: dequantized/decompressed weights.
    """
    decompressed_weight = (
        compressed_weights.astype(TensorDataType.int32) - zero_point if zero_point is not None else compressed_weights
    )
    decompressed_weight = decompressed_weight.astype(scale.dtype) * scale

    if reduction_axis > -1:
        decompressed_weight = ungroup_weights(decompressed_weight, reduction_axis)

    return decompressed_weight


def do_int_quantization(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Performs integer quantization on the given weight tensor.

    :param weight: The weight tensor to quantize.
    :param config: The weight compression configuration.
    :param reduction_axes: Axes along which to reduce (collect) statistics (e.g., min, max). Not required if
        precomputed scale (and zero point) are provided.
    :param precomputed_scale: Optional precomputed scale tensor.
    :param precomputed_zero_point: Optional precomputed zero point tensor.
    :return: A tuple containing the compressed weights, scale, and zero point tensors.
    """
    if not config.is_integer:
        msg = "The function supports integer quantization only"
        raise nncf.InternalError(msg)
    if config.is_asym_mode and (precomputed_scale is None) != (precomputed_zero_point is None):
        msg = (
            "If precomputed quantization parameters are provided, both scale and zero point are required "
            "for asymmetric quantization."
        )
        raise ValueError(msg)

    # When reduction axes are not provided, assuming that the weights are already reshaped
    if config.group_size != -1 and reduction_axes is not None:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)

    # Optimized implementation
    if _can_run_optimized(weight.backend):
        from nncf.openvino.optimized_functions import do_int_quantization as do_int_quantization_ov

        return do_int_quantization_ov(weight, config, reduction_axes, precomputed_scale, precomputed_zero_point)

    # Reference implementation
    if weight.backend == TensorBackend.ov:
        weight = weight.as_numpy_tensor()
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    scale, zero_point = None, None
    if precomputed_scale is None or (config.is_asym_mode and precomputed_zero_point is None):
        if reduction_axes is None:
            msg = "Reduction axes are required for computing the scale and (zero point) vectors."
            raise ValueError(msg)
        scale, zero_point = calculate_integer_quantization_params(weight, reduction_axes, config)
    if precomputed_scale is not None:
        scale = precomputed_scale
    if precomputed_zero_point is not None:
        zero_point = precomputed_zero_point

    compressed_weights = calculate_quantized_weight(weight, config, scale, zero_point)
    return compressed_weights, scale, zero_point


def quantize_dequantize_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
    precomputed_zero_point: Optional[Tensor] = None,
    return_compressed_weight: Optional[bool] = False,
) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    First quantizes the given weight tensor and then dequantizes it back to obtain float32 values.
    :param weight: The weight tensor to quantize-dequantize.
    :param config: Compression configuration.
    :param reduction_axes: Axes along which to reduce (collect) statistics (e.g., min, max). Not required if
        precomputed scale (and zero point) are provided.
    :param precomputed_scale: Optional precomputed scale tensor.
    :param precomputed_zero_point: Optional precomputed zero point tensor.
    :param return_compressed_weight: If True, besides decompressed weight will also return compressed weight, scale,
        (and zero point).
    :return: Dequantized weight tensor or a tuple containing the decompressed weight, compressed weight, scale,
        (and zero point).
    """
    # Optimized implementation
    if _can_run_optimized(weight.backend):
        from nncf.openvino.optimized_functions import quantize_dequantize_weight as quantize_dequantize_weight_ov

        return quantize_dequantize_weight_ov(
            weight,
            config,
            reduction_axes,
            precomputed_scale,
            precomputed_zero_point,
            return_compressed_weight,
        )

    # Reference implementation
    compressed_weight, scale, zero_point = do_int_quantization(
        weight, config, reduction_axes, precomputed_scale, precomputed_zero_point
    )
    decompressed_weight = do_int_dequantization(compressed_weight, scale, zero_point)
    if return_compressed_weight:
        return decompressed_weight, compressed_weight, scale, zero_point
    else:
        return decompressed_weight


def _can_run_optimized(input_backend: TensorBackend) -> bool:
    if (
        input_backend in [TensorBackend.ov, TensorBackend.numpy]
        and os.environ.get("NNCF_DISABLE_OPTIMIZED_COMPRESSION") is None
    ):
        if is_openvino_available():
            from nncf.openvino.cpu_info import is_arm_cpu

            # Due to a bug in CPU plugin compression models can fail at compilation on ARM CPUs. Ticket: 164135.
            return not is_arm_cpu() or is_openvino_at_least("2025.2")
        else:
            nncf_logger.info_once(
                "OpenVINO optimizations are disabled. Install OpenVINO to enable them and improve the performance."
            )
    return False
