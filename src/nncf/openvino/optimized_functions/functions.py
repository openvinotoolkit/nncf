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

from typing import Optional, Union

from nncf import CompressWeightsMode
from nncf.common.utils.caching import disable_results_caching
from nncf.openvino.optimized_functions.models import OV_MODEL_CACHE
from nncf.openvino.optimized_functions.models import OVModelParameters
from nncf.openvino.optimized_functions.models import get_astype_model
from nncf.openvino.optimized_functions.models import get_float_quantization_model
from nncf.openvino.optimized_functions.models import get_float_quantize_dequantize_weight_model
from nncf.openvino.optimized_functions.models import get_integer_quantization_error_model
from nncf.openvino.optimized_functions.models import get_integer_quantization_model
from nncf.openvino.optimized_functions.models import get_integer_quantize_dequantize_weight_model
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorBackend
from nncf.tensor import TensorDataType

ReductionAxes = Union[int, tuple[int, ...]]


def do_integer_quantization(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Tensor = None,
    precomputed_zero_point: Tensor = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Quantizes the given weight tensor to an integer data type.

    :param weight: The weight tensor to quantize.
    :param config: The weight compression configuration.
    :param reduction_axes: Axes along which to reduce (collect) statistics (e.g., min, max). Not required if
        precomputed scale (and zero point) are provided.
    :param precomputed_scale: Optional precomputed scale tensor.
    :param precomputed_zero_point: Optional precomputed zero point tensor.
    :return: A tuple containing the compressed weights, scale, and zero point tensors.
    """
    weight_shape = weight.shape
    scale_shape = None if precomputed_scale is None else precomputed_scale.shape
    zero_point_shape = None if precomputed_zero_point is None else precomputed_zero_point.shape

    ov_model_params = OVModelParameters()
    ov_model_params.input_dtypes["weight"] = weight.dtype
    if precomputed_scale is not None:
        ov_model_params.input_dtypes["scale"] = precomputed_scale.dtype
    if precomputed_zero_point is not None:
        ov_model_params.input_dtypes["zero_point"] = precomputed_zero_point.dtype
    if config.num_bits == 4 and weight.backend == TensorBackend.ov:
        # Return ov tensors in target precision to seamlessly insert them into openvino model later
        ov_model_params.return_ov_tensors = True
        compressed_weight_dtype = TensorDataType.uint4 if config.is_asym_mode else TensorDataType.int4
        ov_model_params.output_dtypes.update(
            {"compressed_weight": compressed_weight_dtype, "zero_point": compressed_weight_dtype}
        )

    model = get_integer_quantization_model(
        ov_model_params,
        config,
        weight_shape,
        scale_shape,
        zero_point_shape,
        reduction_axes,
    )

    if precomputed_scale is None:
        # weight -> compressed_weight, scale, (zero_point)
        results = model([weight])
        if config.is_asym_mode:
            compressed_weight, scale, zero_point = results
        else:
            compressed_weight, scale = results
            zero_point = None

        # Scale is always in fp32 so there is no need to store it in ov.Tensor
        if scale.backend == TensorBackend.ov:
            scale = scale.as_numpy_tensor()
    else:
        # weight, scale, (zero_point) -> compressed_weight
        inputs = (
            [weight, precomputed_scale]
            if precomputed_zero_point is None
            else [weight, precomputed_scale, precomputed_zero_point]
        )
        compressed_weight = model(inputs)[0]
        scale, zero_point = precomputed_scale, precomputed_zero_point

    return compressed_weight, scale, zero_point


def do_float_quantization(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Computes quantization scale if not provided, and performs corresponding float weight quantization.
    NF4 format uses 16 levels in [-1, 1] range, while FP4/MXFP4 uses 16 levels in [-6, 6].

    :param weight: Weight array to compress.
    :param config: Weight compression configuration.
    :param reduction_axes: Axes, along which to reduce (collect) different statistics.
    :param precomputed_scale: Optional precomputed scale.
    :return: Returns quantized weight tensor and corresponding scale tensor.
    """
    assert config.mode in [CompressWeightsMode.NF4, CompressWeightsMode.MXFP4, CompressWeightsMode.FP4]

    weight_shape = weight.shape
    scale_shape = None if precomputed_scale is None else precomputed_scale.shape

    ov_model_params = OVModelParameters()
    ov_model_params.input_dtypes["weight"] = weight.dtype
    if precomputed_scale is not None:
        ov_model_params.input_dtypes["scale"] = precomputed_scale.dtype
    if weight.backend == TensorBackend.ov:
        # Return ov tensors in target precision to seamlessly insert them into openvino model later
        ov_model_params.return_ov_tensors = True
        weight_dtype = TensorDataType.nf4 if config.mode == CompressWeightsMode.NF4 else TensorDataType.f4e2m1
        ov_model_params.output_dtypes.update({"compressed_weight": weight_dtype})

    model = get_float_quantization_model(
        ov_model_params,
        config,
        weight_shape,
        scale_shape,
        reduction_axes,
    )

    if precomputed_scale is None:
        # weight -> compressed_weight, scale
        compressed_weight, scale = model([weight])

        # Scale is always in fp32 so there is no need to store it in ov.Tensor
        if scale.backend == TensorBackend.ov:
            scale = scale.as_numpy_tensor()
    else:
        # weight, scale -> compressed_weight
        compressed_weight = model([weight, precomputed_scale])[0]
        scale = precomputed_scale

    return compressed_weight, scale, None


def integer_quantize_dequantize_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
    precomputed_zero_point: Optional[Tensor] = None,
    return_compressed_weight: Optional[bool] = False,
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """
    Quantizes the given weight tensor to an integer data type and then dequantizes it back to obtain float32 values.

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
    # When reduction axes are not provided, assuming that the weights are already reshaped
    if config.group_size != -1 and reduction_axes is not None:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)

    weight_shape = weight.shape
    scale_shape = precomputed_scale.shape if precomputed_scale is not None else None
    zero_point_shape = precomputed_zero_point.shape if precomputed_zero_point is not None else None

    ov_model_params = OVModelParameters()
    ov_model_params.input_dtypes["weight"] = weight.dtype
    if precomputed_scale is not None:
        ov_model_params.input_dtypes["scale"] = precomputed_scale.dtype
    if precomputed_zero_point is not None:
        ov_model_params.input_dtypes["zero_point"] = precomputed_zero_point.dtype

    model = get_integer_quantize_dequantize_weight_model(
        ov_model_params, config, weight_shape, scale_shape, zero_point_shape, reduction_axes, return_compressed_weight
    )

    inputs = [weight]
    if precomputed_scale is not None:
        inputs.append(precomputed_scale)
    if precomputed_zero_point is not None:
        inputs.append(precomputed_zero_point)

    compressed_weight, scale, zero_point = None, precomputed_scale, precomputed_zero_point
    results = model(inputs)
    if len(results) == 1:
        decompressed_weight = results[0]
    elif len(results) == 2:
        decompressed_weight, compressed_weight = results
    elif len(results) == 3:
        decompressed_weight, compressed_weight, scale = results
    else:
        decompressed_weight, compressed_weight, scale, zero_point = results
    if return_compressed_weight:
        return decompressed_weight, compressed_weight, scale, zero_point
    return decompressed_weight


def float_quantize_dequantize_weight(
    weight: Tensor,
    config: WeightCompressionConfig,
    reduction_axes: Optional[ReductionAxes] = None,
    precomputed_scale: Optional[Tensor] = None,
    return_compressed_weight: Optional[bool] = False,
) -> Union[Tensor, tuple[Tensor, Tensor, Tensor]]:
    """
    First quantizes the given weight tensor and then dequantizes it back to obtain float32 values.

    :param weight: The weight tensor to quantize-dequantize.
    :param config: Compression configuration.
    :param reduction_axes: Axes along which to reduce statistics. Not required if precomputed scale are provided.
    :param precomputed_scale: Optional precomputed scale tensor.
    :param return_compressed_weight: If True, besides decompressed weight will also return compressed weight and scale.
    :return: Dequantized weight tensor or a tuple containing the decompressed weight, compressed weight and scale.
    """
    assert config.mode in [CompressWeightsMode.NF4, CompressWeightsMode.MXFP4, CompressWeightsMode.FP4]

    # When reduction axes are not provided, assuming that the weights are already reshaped
    if config.group_size != -1 and reduction_axes is not None:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)

    weight_shape = weight.shape
    scale_shape = precomputed_scale.shape if precomputed_scale is not None else None

    ov_model_params = OVModelParameters()
    ov_model_params.input_dtypes["weight"] = weight.dtype
    if precomputed_scale is not None:
        ov_model_params.input_dtypes["scale"] = precomputed_scale.dtype

    model = get_float_quantize_dequantize_weight_model(
        ov_model_params, config, weight_shape, scale_shape, reduction_axes, return_compressed_weight
    )

    inputs = [weight]
    if precomputed_scale is not None:
        inputs.append(precomputed_scale)

    compressed_weight, scale = None, precomputed_scale
    results = model(inputs)
    if len(results) == 1:
        decompressed_weight = results[0]
    elif len(results) == 2:
        decompressed_weight, compressed_weight = results
    else:
        decompressed_weight, compressed_weight, scale = results
    if return_compressed_weight:
        return decompressed_weight, compressed_weight, scale
    return decompressed_weight


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
    original_weight_shape = weight.shape
    original_reduction_axes = reduction_axes

    # When reduction axes are not provided, assuming that the weights are already reshaped
    if config.group_size != -1 and reduction_axes is not None:
        # weights are reshaped from [a1, r, a2] to [a1, r//gs, gs, a2]
        weight, reduction_axes = reshape_weight_for_grouped_quantization(weight, reduction_axes, config.group_size)

    ov_model_params = OVModelParameters()
    ov_model_params.input_dtypes["weight"] = weight.dtype
    model = get_integer_quantization_error_model(
        ov_model_params, config, original_weight_shape, weight.shape, original_reduction_axes, reduction_axes
    )

    quantization_error = model([weight])[0].item()

    return quantization_error


def astype(a: Tensor, dtype: TensorDataType) -> Tensor:
    """
    Converts the given tensor to the specified data type. Allows to convert between u4, i4, bf16 data types which are
    not natively supported by numpy. These data types are represented as the following data types in numpy:
        - bfloat16 -> np.float16,
        - uint4 -> uint8,
        - int4 -> int8.
    :param a: Tensor to change data type for.
    :param dtype: Data type to convert to.
    :return: Tensor with the specified data type.
    """
    ov_model_params = OVModelParameters(
        input_dtypes={"input": a.dtype},
        output_dtypes={"output": dtype},
        release_memory=False,
        return_ov_tensors=True,
    )
    with disable_results_caching(OV_MODEL_CACHE):
        model = get_astype_model(ov_model_params, tuple(a.shape))
    return model([Tensor(a)])[0]
