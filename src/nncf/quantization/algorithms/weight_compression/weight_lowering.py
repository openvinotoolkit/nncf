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
from typing import Optional, Union

import nncf
from nncf.common.logging.logger import nncf_logger
from nncf.common.utils.backend import is_openvino_at_least
from nncf.common.utils.backend import is_openvino_available
from nncf.errors import InvalidGroupSizeError
from nncf.errors import UnsupportedModelError
from nncf.parameters import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.constants import CENTER_OF_F4E2M1_QUANTILES
from nncf.quantization.algorithms.weight_compression.constants import CENTER_OF_NF4_QUANTILES
from nncf.quantization.algorithms.weight_compression.constants import F4E2M1_QUANTILES
from nncf.quantization.algorithms.weight_compression.constants import FP_MAX_VALUES
from nncf.quantization.algorithms.weight_compression.constants import NF4_QUANTILES
from nncf.quantization.algorithms.weight_compression.parameters import CompressedWeight
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.tensor import Tensor
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDataType

ReductionAxes = Union[int, tuple[int, ...]]


OPTIMIZED_COMPRESSION_COMPATIBLE_MODES = (
    CompressWeightsMode.INT8_ASYM,
    CompressWeightsMode.INT8_SYM,
    CompressWeightsMode.INT4_ASYM,
    CompressWeightsMode.INT4_SYM,
    CompressWeightsMode.NF4,
    CompressWeightsMode.MXFP4,
    CompressWeightsMode.FP4,
    CompressWeightsMode.FP8_E4M3,
    CompressWeightsMode.MXFP8_E4M3,
)
MIN_INPUT_SIZE_FOR_OPTIMIZED_COMPRESSION = 10000


def get_reduction_channel_size(weight_shape: tuple[int, ...], reduction_axes: ReductionAxes) -> tuple[int, int]:
    """
    Returns the size of the channel dimension for group quantization.

    :param weight_shape: Shape of the weight tensor.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :return: Size of the channel dimension and new reduction axis.
    """
    if isinstance(reduction_axes, tuple) and len(reduction_axes) == 1:
        reduction_axes = reduction_axes[0]
    if not isinstance(reduction_axes, int):
        msg = f"Group-wise quantization expects a single reduction axis, but given: {reduction_axes}."
        raise UnsupportedModelError(msg)
    channel_size = weight_shape[reduction_axes]
    return channel_size, reduction_axes


def reshape_weight_for_grouped_quantization(
    weight: Tensor, reduction_axes: ReductionAxes, group_size: int
) -> tuple[Tensor, int]:
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
    channel_size, reduction_axes = get_reduction_channel_size(weight.shape, reduction_axes)
    if channel_size % group_size != 0:
        msg = f"Channel size {channel_size} should be divisible by size of group {group_size}."
        raise InvalidGroupSizeError(msg)

    num_groups_per_channel = channel_size // group_size
    shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
    shape[reduction_axes : reduction_axes + 1] = (num_groups_per_channel, group_size)
    reshaped_weight = weight.reshape(shape)
    reduction_axes += 1
    return reshaped_weight, reduction_axes


def calculate_float_quantization_params(
    weight: Tensor, reduction_axes: ReductionAxes, config: WeightCompressionConfig
) -> Tensor:
    """
    Calculates the scale for nf4 or mxfp8_e4m3/mxfp4/fp8_e4m3/fp4 quantization.

    :param weight: Weight array to compress.
    :param reduction_axes: Axes along which to reduce (collect) different statistics (e.g., min, max).
    :param config: Weight compression configuration.
    :return: Scale tensor of float32 type for float quantization.
    """
    assert not config.is_integer

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    scale = fns.max(fns.abs(weight), axis=reduction_axes, keepdims=True)
    if config.mode != CompressWeightsMode.NF4:
        if config.compression_dtype in FP_MAX_VALUES:
            max_val = FP_MAX_VALUES[config.compression_dtype]
        else:
            max_val = fns.max(fns.abs(config.get_numpy_codebook()))
        scale = scale / max_val

    # NOTE: adding machine epsilon to avoid division by zero
    eps = fns.finfo(weight).eps
    scale = fns.where(fns.abs(scale) < eps, eps, scale)

    if config.mode in [CompressWeightsMode.MXFP4, CompressWeightsMode.MXFP8_E4M3]:
        # MXFP types are using E8M0 type scale.
        # It can only contain values [2**(-127), 2**(-126), ..., 2**(126), 2**(127)].
        # Here, we quantize each element of the scale to the smallest possible value greater than or equal to
        # the element value to make it possible to convert the float scale value to a FP format without rounding.
        scale = fns.log2(scale)
        scale = fns.ceil(scale)
        scale = fns.clip(scale, -127, 127)
        scale = 2**scale

    return scale


def do_float_dequantization(compressed_weight: Tensor, scale: Tensor, reduction_axis: int = -1) -> Tensor:
    """
    Dequantize the float-quantized weight tensor.

    :param compressed_weight: Tensor with floating-point values.
    :param scale: Scale tensor used for decompression.
    :param reduction_axis: axis along which weights were reshaped for group quantization and will be reshaped back to
        original shapes. If equals to -1, weights are not reshaped, assumed not a group quantization. Defaults to -1.
    :return: Decompressed weight tensor.
    """
    decompressed_weight = compressed_weight * scale
    if reduction_axis != -1:
        decompressed_weight = ungroup_weights(decompressed_weight, reduction_axis)
    return decompressed_weight


def do_float_quantization(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Optional[Tensor]]:
    """
    Computes quantization scale if not provided and performs corresponding weight quantization.
    NF4 format uses 16 levels in [-1, 1] range, MXFP4 uses 16 levels in [-6, 6], and MXFP8_E4M3 uses 256 levels
    in [-448, 448].
    For CODEBOOK quantization currently returns normalized weight without quantization.

    :param weight: Weight array to compress.
    :param config: Weight compression configuration.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics.
    :param precomputed_scale: Optional precomputed scale.
    :return: Returns quantized (for codebook normalized) weight tensor and corresponding scale tensor and optional
        indexes for codebook.
    """
    assert not config.is_integer

    if config.group_size != -1 and reduction_axes is not None:
        # weights are reshaped: [a1, r, a2] -> [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)

    # Optimized implementation
    if _can_run_optimized(weight, config.mode):
        from nncf.openvino.optimized_functions import do_float_quantization as do_float_quantization_ov

        return do_float_quantization_ov(weight, config, reduction_axes, precomputed_scale)

    original_weight_backend = weight.backend
    if weight.backend == TensorBackend.ov:
        weight = weight.as_numpy_tensor()
    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    scale = precomputed_scale
    if scale is None:
        scale = calculate_float_quantization_params(weight, reduction_axes, config)
    norm_weight = _calculate_normalized_weight(weight, scale)
    if config.is_codebook:
        compressed_weight, indexes = _calculate_codebook_quantized_weight(
            norm_weight, quantiles=config.get_numpy_codebook()
        )
        return compressed_weight, scale, indexes

    if original_weight_backend == TensorBackend.ov:
        # Can convert through OpenVINO and return OpenVINO-native low-precision tensor
        compressed_weight = norm_weight.as_openvino_tensor().astype(config.compression_dtype)
    else:
        compressed_weight = _calculate_float_quantized_weight(norm_weight, config.compression_dtype)

    return compressed_weight, scale, None


def float_quantize_dequantize_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
    return_compressed_weight: Optional[bool] = False,
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """
    First quantizes the given weight tensor to float dtype and then dequantizes it back to obtain float32 values.

    :param weight: The weight tensor to quantize-dequantize.
    :param config: Compression configuration.
    :param reduction_axes: Axes along which to reduce statistics. Not required if precomputed scale are provided.
    :param precomputed_scale: Optional precomputed scale tensor.
    :param return_compressed_weight: If True, besides decompressed weight will also return compressed weight and scale.
    :return: Dequantized weight tensor or a tuple containing the decompressed weight, compressed weight and scale.
    """
    # Optimized implementation
    if _can_run_optimized(weight, config.mode):
        from nncf.openvino.optimized_functions import (
            float_quantize_dequantize_weight as float_quantize_dequantize_weight_ov,
        )

        return float_quantize_dequantize_weight_ov(
            weight,
            config,
            reduction_axes,
            precomputed_scale,
            return_compressed_weight,
        )

    # Reference implementation
    compressed_weight, scale, _ = do_float_quantization(weight, config, reduction_axes, precomputed_scale)
    decompressed_weight = do_float_dequantization(compressed_weight, scale)
    if return_compressed_weight:
        return decompressed_weight, compressed_weight, scale
    return decompressed_weight


def calculate_integer_quantization_params(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
) -> tuple[Tensor, Tensor]:
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

    scale = _calculate_signed_scale(weight, reduction_axes, num_bits)
    return scale, None


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
    if _can_run_optimized(weight, config.mode):
        from nncf.openvino.optimized_functions import (
            get_integer_quantization_error as get_integer_quantization_error_ov,
        )

        return get_integer_quantization_error_ov(weight, reduction_axes, config)

    if weight.backend == TensorBackend.ov:
        weight = weight.as_numpy_tensor()
    orig_shape = weight.shape

    if weight.dtype != TensorDataType.float32:
        weight = weight.astype(TensorDataType.float32)

    decompressed_weight = integer_quantize_dequantize_weight(weight, config, reduction_axes)

    decompressed_weight = decompressed_weight.reshape(orig_shape)
    diff = (decompressed_weight - weight) ** 2
    layer_err = fns.mean(diff, axis=reduction_axes)
    val = fns.max(layer_err)
    return val.item()


def compress_weight(
    weight: Tensor,
    reduction_axes: ReductionAxes,
    config: WeightCompressionConfig,
    precomputed_compressed_weight: CompressedWeight = None,
) -> CompressedWeight:
    """
    Compress weight using compression configuration.

    :param weight: The weight to compress.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics (e.g. min, max).
    :param config: Compression configuration.
    :param precomputed_compressed_weight: Contains precomputed scale and zero point.
    :return: The compressed weight and decompression parameters as instance of CompressedWeight
    """
    precomputed_scale, precomputed_zero_point = (
        (precomputed_compressed_weight.scale, precomputed_compressed_weight.zero_point)
        if precomputed_compressed_weight
        else (None, None)
    )

    if not config.is_integer:
        compressed_weight, scale, indexes = do_float_quantization(weight, config, reduction_axes, precomputed_scale)
        if indexes is not None:
            return CompressedWeight(
                indexes,
                scale,
                None,
                config.codebook_values,
            )
        return CompressedWeight(compressed_weight, scale)
    compressed_weight, scale, zero_point = do_integer_quantization(
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


def do_integer_dequantization(
    compressed_weights: Tensor, scale: Tensor, zero_point: Optional[Tensor] = None, reduction_axis: int = -1
) -> Tensor:
    """
    The method dequantizes the given integer weights to float point data type in accordance with the scale and
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


def do_integer_quantization(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
) -> tuple[Tensor, Tensor, Tensor]:
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
    if _can_run_optimized(weight, config.mode):
        from nncf.openvino.optimized_functions import do_integer_quantization as do_integer_quantization_ov

        return do_integer_quantization_ov(weight, config, reduction_axes, precomputed_scale, precomputed_zero_point)

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

    compressed_weights = _calculate_integer_quantized_weight(weight, config, scale, zero_point)
    return compressed_weights, scale, zero_point


def integer_quantize_dequantize_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
    precomputed_zero_point: Optional[Tensor] = None,
    return_compressed_weight: Optional[bool] = False,
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    First quantizes the given weight tensor to integer dtype and then dequantizes it back to obtain float32 values.

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
    if _can_run_optimized(weight, config.mode):
        from nncf.openvino.optimized_functions import (
            integer_quantize_dequantize_weight as integer_quantize_dequantize_weight_ov,
        )

        return integer_quantize_dequantize_weight_ov(
            weight,
            config,
            reduction_axes,
            precomputed_scale,
            precomputed_zero_point,
            return_compressed_weight,
        )

    # Reference implementation
    compressed_weight, scale, zero_point = do_integer_quantization(
        weight, config, reduction_axes, precomputed_scale, precomputed_zero_point
    )
    decompressed_weight = do_integer_dequantization(compressed_weight, scale, zero_point)
    if return_compressed_weight:
        return decompressed_weight, compressed_weight, scale, zero_point
    return decompressed_weight


def _calculate_float_quantized_weight(norm_weight: Tensor, compression_dtype: TensorDataType) -> Tensor:
    """
    Performs float quantization. Look-up table is used to "round" or "quantize" to the closest quant.

    :param norm_weight: Normalized weight tensor to quantize.
    :param compression_dtype: Target floating-point data type for quantization.
    :return: Tensor with floating-point values, where each of them corresponds to 1 out of N quants.
    """
    assert compression_dtype in [TensorDataType.f8e4m3, TensorDataType.f4e2m1, TensorDataType.nf4]

    if compression_dtype == TensorDataType.f8e4m3:
        from nncf.quantization.algorithms.weight_compression.fp8_conversion import fp32_to_fp8e4m3_values

        quantiles_np = fp32_to_fp8e4m3_values(norm_weight.as_numpy_tensor().data)
        return fns.from_numpy(quantiles_np, backend=norm_weight.backend)

    is_nf4 = compression_dtype == TensorDataType.nf4
    quantiles_np = NF4_QUANTILES if is_nf4 else F4E2M1_QUANTILES
    quantile_centers_np = CENTER_OF_NF4_QUANTILES if is_nf4 else CENTER_OF_F4E2M1_QUANTILES
    quantile_centers = fns.from_numpy(quantile_centers_np, backend=norm_weight.backend)
    indexes = fns.searchsorted(quantile_centers, norm_weight)
    quantiles = fns.from_numpy(quantiles_np, backend=indexes.backend)

    if compression_dtype == TensorDataType.f4e2m1:
        # If in-between two quantiles, round to the nearest even quantile.
        shifted_indexes = fns.clip(indexes + 1, 0, quantiles.size - 1)
        dist_left = fns.abs(norm_weight - quantiles[indexes])
        dist_right = fns.abs(norm_weight - quantiles[shifted_indexes])
        choose_right = (dist_right < dist_left) | ((dist_left == dist_right) & ((shifted_indexes + 1) % 2 == 0))
        indexes = fns.where(choose_right, shifted_indexes, indexes)

    quantized_weight = quantiles[indexes]
    return quantized_weight


def _calculate_codebook_quantized_weight(
    norm_weight: Tensor, quantiles: Tensor = None, center_of_quantiles: Tensor = None
) -> tuple[Tensor, Tensor]:
    """
    Performs quantization by quantiles (if center_of_quantiles is None). Look-up table is used to
    "round" or "quantize" to the closest quant.

    :param norm_weight: Weight tensor to quantize already normalized to quantiles range.
    :param quantiles: Quantiles to use for quantization. If None, the center_of_quantiles must be provided.
    :param center_of_quantiles: Center of quantiles to use for quantization. If None, it is calculated as the average
        of adjacent quantiles.
    :return: Tensor with floating-point values, where each of them corresponds to elements from quantiles.
    """
    assert quantiles is not None or center_of_quantiles is not None, (
        "Either quantiles or center_of_quantiles should be provided"
    )

    if center_of_quantiles is None:
        center_of_quantiles = 0.5 * (quantiles[1:] + quantiles[:-1])
    center_of_quantiles = fns.from_numpy(center_of_quantiles, backend=norm_weight.backend)
    indexes = fns.searchsorted(center_of_quantiles, norm_weight)
    quantiles = fns.from_numpy(quantiles, backend=indexes.backend)
    quantized_weight = quantiles[indexes]
    return quantized_weight, indexes


def _calculate_normalized_weight(weight: Tensor, scale: Tensor) -> Tensor:
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


def _calculate_signed_scale(weight: Tensor, reduction_axes: ReductionAxes, num_bits=4) -> Tensor:
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


def _calculate_integer_quantized_weight(
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


def _can_run_optimized(inp: Tensor, mode: CompressWeightsMode) -> bool:
    if (
        inp.backend in [TensorBackend.ov, TensorBackend.numpy]
        and inp.size >= MIN_INPUT_SIZE_FOR_OPTIMIZED_COMPRESSION
        and os.environ.get("NNCF_DISABLE_OPTIMIZED_COMPRESSION") is None
        and mode in OPTIMIZED_COMPRESSION_COMPATIBLE_MODES
    ):
        if is_openvino_available():
            from nncf.openvino.cpu_info import is_arm_cpu

            # Due to a bug in CPU plugin compression models can fail at compilation on ARM CPUs. Ticket: 164135.
            return not is_arm_cpu() or is_openvino_at_least("2025.2")
        nncf_logger.info_once(
            "OpenVINO optimizations are disabled. Install OpenVINO to enable them and improve the performance."
        )
    return False
