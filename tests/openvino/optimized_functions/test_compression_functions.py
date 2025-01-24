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

from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from unittest.mock import patch

import numpy as np
import openvino as ov
import pytest

import nncf.openvino.optimized_functions as opt_fns
from nncf import CompressWeightsMode
from nncf.common.utils.caching import ResultsCache
from nncf.common.utils.caching import cache_results
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_int_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import reshape_weight_for_grouped_quantization
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP as DTYPE_MAP_NP
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP_REV as DTYPE_MAP_REV_NP
from nncf.tensor.functions.openvino_numeric import DTYPE_MAP as DTYPE_MAP_OV


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

REDUCTION_AXES = (1,)

RANDOM_TENSOR_CACHE_CONTAINER = ResultsCache()


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
        data = data.as_numpy_tensor() if dtype == TensorDataType.bfloat16 else Tensor(data)
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
    import nncf.common.utils.backend

    original_value = nncf.common.utils.backend._OPENVINO_AVAILABLE
    nncf.common.utils.backend._OPENVINO_AVAILABLE = available
    yield
    nncf.common.utils.backend._OPENVINO_AVAILABLE = original_value


@pytest.mark.parametrize("weight_shape", [(100000, 4)], ids=[""])
@pytest.mark.parametrize("config", COMPRESSION_CONFIGS, ids=[str(c) for c in COMPRESSION_CONFIGS])
@pytest.mark.parametrize(
    ("quantization_task", "tensor_backend"),
    [
        (QuantizationTask.Q, TensorBackend.numpy),
        (QuantizationTask.Q, "auto"),
        # NumPy backend should support OV tensors as inputs only for quantization task
        (QuantizationTask.Q, TensorBackend.ov),
        (QuantizationTask.Q_DQ, TensorBackend.numpy),
        (QuantizationTask.Q_DQ, "auto"),
        (QuantizationTask.Q_DQ_RQ, TensorBackend.numpy),
        (QuantizationTask.Q_DQ_RQ, "auto"),
    ],
)
@pytest.mark.parametrize("dtype", [TensorDataType.float32, TensorDataType.float16, TensorDataType.bfloat16])
@pytest.mark.parametrize("precompute_s_zp", [False, True], ids=["no-precompute", "precompute"])
def test_quantization_alignment(weight_shape, config, quantization_task, tensor_backend, dtype, precompute_s_zp):
    d1, d2 = weight_shape
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
            weight = get_random_float_tensor(weight_shape, dtype, weight_tensor_backend)
            precomputed_scale, precomputed_zero_point = None, None
            if precompute_s_zp:
                # When scale (and z.p) are precomputed, all inputs are assumed to be reshaped beforehand
                if group_size != -1:
                    weight, _ = reshape_weight_for_grouped_quantization(weight, REDUCTION_AXES, group_size)

                precomputed_scale = get_random_float_tensor(scale_shape, TensorDataType.float32, TensorBackend.numpy)
                if config.is_asym_mode:
                    precomputed_zero_point = get_random_integer_tensor(
                        zero_point_shape, level_low, level_high, TensorDataType.int32, TensorBackend.numpy
                    )

            if quantization_task == QuantizationTask.Q:
                fn_to_call = do_int_quantization
                fn_to_patch = opt_fns.do_int_quantization
            else:
                fn_to_call = quantize_dequantize_weight
                fn_to_patch = opt_fns.quantize_dequantize_weight
            patch_path = f"nncf.openvino.optimized_functions.{fn_to_patch.__name__}"
            with patch(patch_path, side_effect=fn_to_patch) as mock:
                # When scale (and z.p) are precomputed, all inputs are assumed to be already reshaped and reduction
                # axes are not needed
                reduction_axes = None if precompute_s_zp else REDUCTION_AXES

                kwargs = {}
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

        if quantization_task != QuantizationTask.Q_DQ and precompute_s_zp:
            # In case of precomputed scale or zero point, the returned scale and z.p. should equal the given ones
            np.testing.assert_allclose(precomputed_scale.data, scale.data, atol=0, rtol=0)
            if config.is_asym_mode:
                np.testing.assert_allclose(precomputed_zero_point.data, zero_point.data, atol=0, rtol=0)

        # Save results for comparison between implementations
        if quantization_task != QuantizationTask.Q:
            results[cb]["decompressed_weight"] = decompressed_weight
        if quantization_task != QuantizationTask.Q_DQ:
            results[cb]["compressed_weight"] = compressed_weight.as_numpy_tensor()
            results[cb]["scale"] = scale
            if config.is_asym_mode:
                results[cb]["zero_point"] = zero_point.as_numpy_tensor()

        _check_backends_and_dtypes(
            quantization_task,
            cb,
            weight_tensor_backend,
            config,
            precompute_s_zp,
            compressed_weight,
            scale,
            zero_point,
            decompressed_weight,
        )

    _check_values(results)


def _check_backends_and_dtypes(
    quantization_task,
    cb,
    weight_tensor_backend,
    config,
    precompute_s_zp,
    compressed_weight,
    scale,
    zero_point,
    decompressed_weight,
):
    if quantization_task != QuantizationTask.Q_DQ:
        # Scale should always be float32 and numpy backend
        assert scale.dtype == TensorDataType.float32
        assert scale.backend == TensorBackend.numpy

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
        assert compressed_weight.dtype == (TensorDataType.uint4 if config.is_asym_mode else TensorDataType.int4)
        if config.is_asym_mode and not precompute_s_zp:
            assert zero_point.backend == TensorBackend.ov
            assert zero_point.dtype == TensorDataType.uint4
    else:
        if quantization_task != QuantizationTask.Q_DQ:
            # Otherwise compressed weight and zero point must be returned in numpy backend, compressed weight must
            # be of (u)int8 data type, zero point -- in int32
            assert compressed_weight.backend == TensorBackend.numpy
            assert compressed_weight.dtype == (TensorDataType.uint8 if config.is_asym_mode else TensorDataType.int8)
            if config.is_asym_mode and not precompute_s_zp:
                assert zero_point.backend == TensorBackend.numpy
                assert zero_point.dtype == TensorDataType.int32
        if quantization_task != QuantizationTask.Q:
            assert decompressed_weight.backend == TensorBackend.numpy
            assert decompressed_weight.dtype == TensorDataType.float32


def _check_values(results):
    # Check that the computed tensors are equal between implementations
    keys = set(results[ComputationBackend.OV]).union(set(results[ComputationBackend.NumPy]))
    for key in keys:
        numpy_result = results[ComputationBackend.NumPy][key]
        ov_result = results[ComputationBackend.OV][key]

        # Note: For static-shaped OV models doing asymmetric compression with convertable divisions there maybe
        # misalignments equal to 1 quant between OV and NumPy. For more details see ticket 156511.

        np.testing.assert_allclose(
            ov_result.data, numpy_result.data, atol=0, rtol=0, err_msg=f"Results do not align for {key}."
        )
