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
import inspect
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from unittest.mock import patch

import numpy as np
import openvino as ov
import pytest

from nncf import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.openvino_modeling import OVModelParameters
from nncf.quantization.algorithms.weight_compression.openvino_modeling import get_compress_decompress_weight_model
from nncf.quantization.algorithms.weight_compression.openvino_modeling import get_compress_weight_model
from nncf.quantization.algorithms.weight_compression.weight_lowering import calculate_quantized_dequantized_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.results_caching import ResultsCacheContainer
from nncf.results_caching import cache_results
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import functions as fns
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP as DTYPE_MAP_NP
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP_REV as DTYPE_MAP_REV_NP
from nncf.tensor.functions.ov import DTYPE_MAP as DTYPE_MAP_OV


class ComputationBackend(Enum):
    NumPy = "numpy"
    OV = "ov"


class QuantizationTask(Enum):
    Q = "quantize"
    Q_DQ = "quantize_dequantize"
    Q_DQ_RQ = "quantize_dequantize_return_quantized"


COMPRESSION_CONFIGS = [
    WeightCompressionConfig(CompressWeightsMode.INT8_ASYM),
    WeightCompressionConfig(CompressWeightsMode.INT8_SYM),
    WeightCompressionConfig(CompressWeightsMode.INT4_ASYM),
    WeightCompressionConfig(CompressWeightsMode.INT4_SYM),
    WeightCompressionConfig(CompressWeightsMode.INT4_ASYM, group_size=2),
    WeightCompressionConfig(CompressWeightsMode.INT4_SYM, group_size=2),
]


DATA_TYPES = [TensorDataType.float32, TensorDataType.float16, TensorDataType.bfloat16]

WEIGHT_SHAPE = (10000, 4)

MAX_MISALIGNMENT_FREQUENCY = {
    TensorDataType.float32: 1e-2,  # tends to < 5e-6
    TensorDataType.float16: 1e-2,  # tends to < 5e-5
    TensorDataType.bfloat16: 1e-2,  # tends to < 5e-4
}

MAX_MISALIGNMENT_MAGNITUDE = 1

TENSOR_BACKENDS = [TensorBackend.numpy, TensorBackend.ov]

EPS = np.finfo(np.float32).eps

REDUCTION_AXES = (1,)


RANDOM_TENSOR_CACHE_CONTAINER = ResultsCacheContainer()


@cache_results(RANDOM_TENSOR_CACHE_CONTAINER)
def get_random_float_tensor(shape, dtype, backend, seed=0):
    np.random.seed(seed)
    data = np.random.normal(size=shape)
    data = data.astype(np.float16 if dtype == TensorDataType.float16 else np.float32)

    if backend == TensorBackend.ov or dtype == TensorDataType.bfloat16:
        data = Tensor(ov.Tensor(data, shape, DTYPE_MAP_OV[DTYPE_MAP_REV_NP[data.dtype]]))
        if dtype == TensorDataType.bfloat16:
            data = data.astype(TensorDataType.bfloat16)
    if backend == TensorBackend.numpy:
        data = data.to_backend(TensorBackend.numpy) if dtype == TensorDataType.bfloat16 else Tensor(data)
    return Tensor(data)


@cache_results(RANDOM_TENSOR_CACHE_CONTAINER)
def get_random_integer_tensor(shape, low, high, dtype, backend, seed=0):
    np.random.seed(seed)
    data = np.random.randint(low, high, size=shape).astype(DTYPE_MAP_NP[dtype])
    if backend == TensorBackend.ov:
        data = ov.Tensor(data, shape, DTYPE_MAP_OV[dtype])
    return Tensor(data)


@contextmanager
def openvino_available(available: bool):
    import nncf.utils

    original_value = nncf.utils._openvino_available
    nncf.utils._openvino_available = available
    yield
    nncf.utils._openvino_available = original_value


@pytest.mark.parametrize("config", COMPRESSION_CONFIGS, ids=[str(c) for c in COMPRESSION_CONFIGS])
@pytest.mark.parametrize(
    ("quantization_task", "tensor_backend"),
    [
        (QuantizationTask.Q, TensorBackend.numpy),
        (QuantizationTask.Q, "auto"),
        # Only for quantization task NumPy backend should support OV tensors as inputs
        (QuantizationTask.Q, TensorBackend.ov),
        (QuantizationTask.Q_DQ, TensorBackend.numpy),
        (QuantizationTask.Q_DQ, "auto"),
        (QuantizationTask.Q_DQ_RQ, TensorBackend.numpy),
        (QuantizationTask.Q_DQ_RQ, "auto"),
    ],
)
@pytest.mark.parametrize("dtype", DATA_TYPES)
@pytest.mark.parametrize("precompute_s_zp", [False, True], ids=["no-precompute", "precompute"])
@pytest.mark.parametrize("static_shapes", [False, True], ids=["dynamic-shapes", "static-shapes"])
def test_quantization_alignment(config, quantization_task, tensor_backend, dtype, precompute_s_zp, static_shapes):
    d1, d2 = WEIGHT_SHAPE
    group_size = config.group_size
    zero_point_shape = scale_shape = (d1, 1) if group_size == -1 else (d1, d2 // group_size, 1)
    level_low, level_high = 0, 2**config.num_bits - 1

    results = defaultdict(dict)
    # Iterate over two implementations
    for cb in [ComputationBackend.NumPy, ComputationBackend.OV]:
        # A context manager to enable/disable ov implementation
        with openvino_available(cb == ComputationBackend.OV):
            # OV tensor backend for weight is only supported for quantization task
            if quantization_task == QuantizationTask.Q and (
                tensor_backend == TensorBackend.ov or cb == ComputationBackend.OV and tensor_backend == "auto"
            ):
                weight_tensor_backend = TensorBackend.ov
            else:
                weight_tensor_backend = TensorBackend.numpy

            # Generate input tensors
            weight = get_random_float_tensor(WEIGHT_SHAPE, dtype, weight_tensor_backend)
            precomputed_scale, precomputed_zero_point = None, None
            if precompute_s_zp:
                # When scale (and z.p) are precomputed, all inputs are assumed to be reshaped beforehand
                if group_size != -1:
                    weight, _ = reshape_weight_for_grouped_quantization(weight, REDUCTION_AXES, group_size)

                precomputed_scale = get_random_float_tensor(scale_shape, TensorDataType.float32, TensorBackend.numpy)
                if config.is_int_asym:
                    precomputed_zero_point = get_random_integer_tensor(
                        zero_point_shape, level_low, level_high, TensorDataType.int32, TensorBackend.numpy
                    )

            if quantization_task == QuantizationTask.Q:
                fn_to_call = do_int_quantization
                fn_to_patch = get_compress_weight_model
            else:
                fn_to_call = calculate_quantized_dequantized_weight
                fn_to_patch = get_compress_decompress_weight_model
            patch_path = f"{inspect.getmodule(fn_to_patch).__name__}.{fn_to_patch.__name__}"
            with patch(patch_path, side_effect=fn_to_patch) as mock:
                # When scale (and z.p) are precomputed, all inputs are assumed to be already reshaped and reduction
                # axes are not needed
                reduction_axes = None if precompute_s_zp else REDUCTION_AXES

                kwargs = {}
                if cb == ComputationBackend.OV:
                    ov_model_params = OVModelParameters(weight.dtype, dynamic_shapes=not static_shapes)
                    kwargs["ov_model_params"] = ov_model_params
                if quantization_task == QuantizationTask.Q_DQ_RQ:
                    kwargs["return_compressed_weight"] = True

                outputs = fn_to_call(
                    weight, config, reduction_axes, precomputed_scale, precomputed_zero_point, **kwargs
                )

                decompressed_weight, compressed_weight, scale, zero_point = (None,) * 4
                if quantization_task == QuantizationTask.Q:
                    compressed_weight, scale, zero_point = outputs
                elif quantization_task == QuantizationTask.Q_DQ:
                    decompressed_weight = outputs
                else:
                    decompressed_weight, compressed_weight, scale, zero_point = outputs

                if cb == ComputationBackend.NumPy:
                    mock.assert_not_called()
                else:
                    mock.assert_called_once()

        if quantization_task != QuantizationTask.Q_DQ:
            # Scale should always be float32 and numpy backend
            assert scale.dtype == TensorDataType.float32
            assert scale.backend == TensorBackend.numpy
            if precompute_s_zp:
                # In case of precomputed scale or zero point, the returned scale and z.p. should equal the given ones
                np.testing.assert_allclose(precomputed_scale.data, scale.data)
                if config.is_int_asym:
                    np.testing.assert_allclose(precomputed_zero_point.data, zero_point.data)

        if (
            quantization_task == QuantizationTask.Q
            and cb == ComputationBackend.OV
            and weight_tensor_backend == TensorBackend.ov
            and config.num_bits == 4
        ):
            # For 4 bit compression in case of ov implementation and ov backend the compressed weight and the computed
            # zero point must be in ov backend and have (u)int4 dtype in order to be able to insert them into OV model
            # without re-packing
            assert compressed_weight.backend == TensorBackend.ov
            assert compressed_weight.dtype == (TensorDataType.uint4 if config.is_int_asym else TensorDataType.int4)
            if config.is_int_asym and not precompute_s_zp:
                assert zero_point.backend == TensorBackend.ov
                assert zero_point.dtype == TensorDataType.uint4
        else:
            if quantization_task != QuantizationTask.Q_DQ:
                # Otherwise compressed weight and zero point must be returned in numpy backend, compressed weight must
                # be of (u)int8 data type, zero point -- in int32
                assert compressed_weight.backend == TensorBackend.numpy
                assert compressed_weight.dtype == (TensorDataType.uint8 if config.is_int_asym else TensorDataType.int8)
                if config.is_int_asym and not precompute_s_zp:
                    assert zero_point.backend == TensorBackend.numpy
                    assert zero_point.dtype == TensorDataType.int32
            if quantization_task != QuantizationTask.Q:
                assert decompressed_weight.backend == TensorBackend.numpy
                assert decompressed_weight.dtype == TensorDataType.float32

        # Save results for comparison between implementations
        if quantization_task != QuantizationTask.Q:
            results[cb]["decompressed_weight"] = decompressed_weight
        if quantization_task != QuantizationTask.Q_DQ:
            results[cb]["compressed_weight"] = compressed_weight.to_backend(TensorBackend.numpy)
            results[cb]["scale"] = scale
            if config.is_int_asym:
                results[cb]["zero_point"] = zero_point.to_backend(TensorBackend.numpy)

    keys = set(results[ComputationBackend.OV]).union(set(results[ComputationBackend.NumPy]))
    for key in keys:
        numpy_result = results[ComputationBackend.NumPy][key]
        ov_result = results[ComputationBackend.OV][key]

        atol = 0
        scale = None
        # For static-shaped OV models doing asymmetric compression there maybe misalignments between OV and NumPy
        # For more details see 156511
        if static_shapes and config.is_int_asym:
            if key == "compressed_weight":
                atol = MAX_MISALIGNMENT_MAGNITUDE
            elif key == "decompressed_weight":
                if "scale" in results[ComputationBackend.NumPy]:
                    scale = results[ComputationBackend.NumPy]["scale"]
                else:
                    if precompute_s_zp:
                        scale = precomputed_scale
                    else:
                        weight = get_random_float_tensor(WEIGHT_SHAPE, dtype, TensorBackend.numpy)
                        with openvino_available(False):
                            _, _, scale, _ = calculate_quantized_dequantized_weight(
                                weight, config, REDUCTION_AXES, return_compressed_weight=True
                            )
                # For decompressed weight the misalignment magnitude depends on the scale
                atol = MAX_MISALIGNMENT_MAGNITUDE * fns.abs(scale).max().item() + EPS
            max_misalignment_frequency = MAX_MISALIGNMENT_FREQUENCY[dtype]
        else:
            max_misalignment_frequency = None

        # Check that the computed tensors are equal between implementations
        np.testing.assert_allclose(
            numpy_result.data, ov_result.data, atol=atol, err_msg=f"Results do not align for {key}."
        )

        if max_misalignment_frequency is not None:
            if key == "compressed_weight":
                diff = fns.abs(numpy_result.astype(TensorDataType.int32) - ov_result.astype(TensorDataType.int32))
            else:
                diff = fns.abs(numpy_result - ov_result)

            if diff.max() > 0:
                # Check that the proportion of misaligned values is small
                n_not_equal = fns.sum(diff > 0)
                assert n_not_equal / numpy_result.size < max_misalignment_frequency

                # Check that the magnitude of misalignment is as small as expected
                if key == "decompressed_weight":
                    # Reshape scale to match the shape of decompressed weight
                    scale = np.repeat(scale.data, diff.shape[-1], axis=-1)
                    np.testing.assert_array_less(
                        diff.data,
                        MAX_MISALIGNMENT_MAGNITUDE * np.abs(scale) + EPS,
                        err_msg=f"Too large misalignment for {key}.",
                    )


# @pytest.mark.parametrize("mode", COMPRESSION_MODES)
# @pytest.mark.parametrize("group_size", [2])
# def test_grouped_quantization(mode, group_size):
#     if mode in [CompressWeightsMode.INT8_SYM, CompressWeightsMode.INT8_ASYM]:
#         pytest.skip("Group size is not applicable for INT8 modes")
#
#     # Generate random weight tensor
#     weight_shape = (128, 4)
#     weight = get_random_float_tensor(weight_shape, TensorDataType.float32, TensorBackend.numpy)
#
#     # Create WeightCompressionConfig
#     config = WeightCompressionConfig(mode, group_size=group_size)
#
#     # Patch is_openvino_available to control the implementation
#     with patch("nncf.utils.is_openvino_available", return_value=False):
#         # Reference implementation
#         decompressed_weight_ref = calculate_quantized_dequantized_weight(weight, config, REDUCTION_AXES)
#
#     with patch("nncf.utils.is_openvino_available", return_value=True):
#         # OpenVINO implementation
#         ov_model_params = OVModelParameters(weight.dtype)
#         decompressed_weight_ov = calculate_quantized_dequantized_weight(
#             weight, config, REDUCTION_AXES, ov_model_params=ov_model_params
#         )
#
#     # Compare decompressed weights
#     np.testing.assert_allclose(decompressed_weight_ref.data, decompressed_weight_ov.data, atol=1e-5, rtol=1e-4)
#
#
# def test_weight_dtypes():
#     # Test different weight data types
#     weight_shape = (128, 4)
#     for dtype in DATA_TYPES:
#         weight = get_random_float_tensor(weight_shape, dtype, TensorBackend.numpy)
#         config = WeightCompressionConfig(CompressWeightsMode.INT8_SYM)
#
#         # Reference implementation
#         with patch("nncf.utils.is_openvino_available", return_value=False):
#             decompressed_weight_ref = calculate_quantized_dequantized_weight(weight, config, REDUCTION_AXES)
#
#         # OpenVINO implementation
#         with patch("nncf.utils.is_openvino_available", return_value=True):
#             ov_model_params = OVModelParameters(weight.dtype)
#             decompressed_weight_ov = calculate_quantized_dequantized_weight(
#                 weight, config, REDUCTION_AXES, ov_model_params=ov_model_params
#             )
#
#         # Compare decompressed weights
#         np.testing.assert_allclose(decompressed_weight_ref.data, decompressed_weight_ov.data, atol=1e-5, rtol=1e-4)
#
#
# def test_tensor_backends():
#     # Test different tensor backends for do_int_quantization
#     weight_shape = (128, 4)
#     weight_numpy = get_random_float_tensor(weight_shape, TensorDataType.float32, TensorBackend.numpy)
#     weight_ov = get_random_float_tensor(weight_shape, TensorDataType.float32, TensorBackend.ov)
#     config = WeightCompressionConfig(CompressWeightsMode.INT8_SYM)
#
#     # Reference implementation with numpy backend
#     with patch("nncf.utils.is_openvino_available", return_value=False):
#         compressed_weight_ref, scale_ref = do_int_quantization(weight_numpy, config, REDUCTION_AXES)
#
#     # OpenVINO implementation with OV backend
#     with patch("nncf.utils.is_openvino_available", return_value=True):
#         ov_model_params = OVModelParameters(weight_ov.dtype)
#         compressed_weight_ov, scale_ov = do_int_quantization(
#             weight_ov, config, REDUCTION_AXES, ov_model_params=ov_model_params
#         )
#
#     # Compare compressed weights
#     np.testing.assert_allclose(compressed_weight_ref.data, compressed_weight_ov.data, atol=1e-5, rtol=1e-4)
#
#     # Compare scales
#     np.testing.assert_allclose(scale_ref.data, scale_ov.data, atol=1e-5, rtol=1e-4)