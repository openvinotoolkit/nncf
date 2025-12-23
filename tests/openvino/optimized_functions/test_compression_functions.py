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
import re
from collections import defaultdict
from contextlib import contextmanager
from enum import Enum
from unittest.mock import patch

import numpy as np
import openvino as ov
import openvino.opset13 as opset
import pytest

import nncf
import nncf.openvino.optimized_functions as opt_fns
from nncf import CompressWeightsMode
from nncf import Dataset
from nncf.common.factory import NNCFGraphFactory
from nncf.common.utils.caching import ResultsCache
from nncf.common.utils.caching import cache_results
from nncf.openvino.cpu_info import is_arm_cpu
from nncf.openvino.graph.node_utils import get_const_value_as_ov_tensor
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import MIN_INPUT_SIZE_FOR_OPTIMIZED_COMPRESSION
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_float_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import float_quantize_dequantize_weight
from nncf.quantization.algorithms.weight_compression.weight_lowering import get_integer_quantization_error
from nncf.quantization.algorithms.weight_compression.weight_lowering import integer_quantize_dequantize_weight
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


INT8_COMPRESSION_CONFIGS = [
    WeightCompressionConfig(CompressWeightsMode.INT8_ASYM),
    WeightCompressionConfig(CompressWeightsMode.INT8_SYM),
]

INT4_COMPRESSION_CONFIGS = [
    WeightCompressionConfig(CompressWeightsMode.INT4_ASYM),
    WeightCompressionConfig(CompressWeightsMode.INT4_SYM),
    WeightCompressionConfig(CompressWeightsMode.INT4_ASYM, group_size=2),
    WeightCompressionConfig(CompressWeightsMode.INT4_SYM, group_size=2),
]

FP4_COMPRESSION_CONFIGS = [
    WeightCompressionConfig(CompressWeightsMode.NF4),
    WeightCompressionConfig(CompressWeightsMode.FP4),
    WeightCompressionConfig(CompressWeightsMode.NF4, group_size=2),
    WeightCompressionConfig(CompressWeightsMode.FP4, group_size=2),
    WeightCompressionConfig(CompressWeightsMode.MXFP4, group_size=32),
]

FP8_COMPRESSION_CONFIGS = [
    WeightCompressionConfig(CompressWeightsMode.FP8_E4M3),
    WeightCompressionConfig(CompressWeightsMode.FP8_E4M3, group_size=2),
    WeightCompressionConfig(CompressWeightsMode.MXFP8_E4M3, group_size=32),
]

COMPRESSION_CONFIGS = (
    INT8_COMPRESSION_CONFIGS + INT4_COMPRESSION_CONFIGS + FP4_COMPRESSION_CONFIGS + FP8_COMPRESSION_CONFIGS
)

WEIGHT_SHAPE = (10000, 32)

REDUCTION_AXES = (1,)

RANDOM_TENSOR_CACHE_CONTAINER = ResultsCache()

SUPPORTED_WEIGHT_DTYPES = [
    TensorDataType.float32,
    TensorDataType.float16,
    TensorDataType.bfloat16,
    TensorDataType.f8e4m3,
    TensorDataType.f8e5m2,
]


@cache_results(RANDOM_TENSOR_CACHE_CONTAINER)
def get_random_float_tensor(shape, dtype, backend, seed=0):
    np.random.seed(seed)
    data = np.random.normal(size=shape)
    data = data.astype(np.float16 if dtype == TensorDataType.float16 else np.float32)

    unsupported_dtype_in_numpy = dtype in [TensorDataType.bfloat16, TensorDataType.f8e5m2, TensorDataType.f8e4m3]
    if backend == TensorBackend.ov or unsupported_dtype_in_numpy:
        data = Tensor(ov.Tensor(data, shape, DTYPE_MAP_OV[DTYPE_MAP_REV_NP[data.dtype]]))
        if unsupported_dtype_in_numpy:
            data = data.astype(dtype)
    if backend == TensorBackend.numpy:
        data = data.as_numpy_tensor() if unsupported_dtype_in_numpy else Tensor(data)
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
    import nncf.quantization.algorithms.weight_compression.weight_lowering as lowering

    with patch.object(lowering, "_can_run_optimized", return_value=available):
        yield


@pytest.mark.parametrize(
    "weight_shape,is_disabled",
    [
        ((MIN_INPUT_SIZE_FOR_OPTIMIZED_COMPRESSION // 4 - 1, 4), True),
        ((MIN_INPUT_SIZE_FOR_OPTIMIZED_COMPRESSION // 4, 4), False),
    ],
)
@pytest.mark.parametrize("quantization_task", [QuantizationTask.Q, QuantizationTask.Q_DQ, QuantizationTask.Q_DQ_RQ])
def test_optimized_compression_is_disabled(weight_shape, is_disabled, quantization_task):
    weight = get_random_float_tensor(weight_shape, TensorDataType.float32, TensorBackend.numpy)
    config = WeightCompressionConfig(CompressWeightsMode.INT8_ASYM)

    fn_to_call, fn_to_patch = _get_compression_fn_from_quantization_task(quantization_task, config)
    patch_path = f"nncf.openvino.optimized_functions.{fn_to_patch.__name__}"
    with patch(patch_path, side_effect=fn_to_patch) as mock:
        kwargs = {}
        if quantization_task == QuantizationTask.Q_DQ_RQ:
            kwargs["return_compressed_weight"] = True

        fn_to_call(weight, config, reduction_axes=1)

        if is_disabled:
            mock.assert_not_called()
        else:
            mock.assert_called_once()


@pytest.mark.parametrize("weight_shape", [WEIGHT_SHAPE], ids=[""])
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
@pytest.mark.parametrize("dtype", SUPPORTED_WEIGHT_DTYPES)
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

            fn_to_call, fn_to_patch = _get_compression_fn_from_quantization_task(quantization_task, config)
            patch_path = f"nncf.openvino.optimized_functions.{fn_to_patch.__name__}"
            with patch(patch_path, side_effect=fn_to_patch) as mock:
                # When scale (and z.p) are precomputed, all inputs are assumed to be already reshaped and reduction
                # axes are not needed
                reduction_axes = None if precompute_s_zp else REDUCTION_AXES

                kwargs = {}
                if quantization_task == QuantizationTask.Q_DQ_RQ:
                    kwargs["return_compressed_weight"] = True

                args = (weight, config, reduction_axes, precomputed_scale)
                if config.is_integer:
                    args = args + (precomputed_zero_point,)
                outputs = fn_to_call(*args, **kwargs)

                decompressed_weight, compressed_weight, scale, zero_point = (None,) * 4
                if quantization_task == QuantizationTask.Q:
                    if config.is_integer:
                        compressed_weight, scale, zero_point = outputs
                    else:
                        compressed_weight, scale, _ = outputs
                elif quantization_task == QuantizationTask.Q_DQ:
                    decompressed_weight = outputs
                else:
                    if config.is_integer:
                        decompressed_weight, compressed_weight, scale, zero_point = outputs
                    else:
                        decompressed_weight, compressed_weight, scale = outputs

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
        if group_size != -1 and not precompute_s_zp:
            weight, _ = reshape_weight_for_grouped_quantization(weight, REDUCTION_AXES, group_size)
        results[cb]["input"] = weight.as_numpy_tensor()

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


@pytest.mark.parametrize("weight_shape", [WEIGHT_SHAPE], ids=[""])
@pytest.mark.parametrize("config", INT4_COMPRESSION_CONFIGS, ids=[str(c) for c in INT4_COMPRESSION_CONFIGS])
@pytest.mark.parametrize("tensor_backend", [TensorBackend.numpy, "auto"])
@pytest.mark.parametrize("dtype", SUPPORTED_WEIGHT_DTYPES)
@pytest.mark.parametrize("reduction", ["max_mean", "frobenius"])
def test_integer_quantization_error_alignment(weight_shape, config, tensor_backend, dtype, reduction):
    results = defaultdict(dict)
    # Iterate over two implementations
    for cb in [ComputationBackend.NumPy, ComputationBackend.OV]:
        # A context manager to enable/disable ov implementation
        with openvino_available(cb == ComputationBackend.OV):
            if tensor_backend == TensorBackend.ov or cb == ComputationBackend.OV and tensor_backend == "auto":
                weight_tensor_backend = TensorBackend.ov
            else:
                weight_tensor_backend = TensorBackend.numpy

            weight = get_random_float_tensor(weight_shape, dtype, weight_tensor_backend)
            fn_to_patch = opt_fns.get_integer_quantization_error
            patch_path = f"nncf.openvino.optimized_functions.{fn_to_patch.__name__}"
            with patch(patch_path, side_effect=fn_to_patch) as mock:
                results[cb]["quantization_error"] = get_integer_quantization_error(
                    weight, REDUCTION_AXES, config, reduction=reduction
                )

            if cb == ComputationBackend.NumPy:
                mock.assert_not_called()
            else:
                mock.assert_called_once()

    # For "max_mean", it seems like numpy and openvino summate elements in different order during
    # reduce_sum / reduce_mean computation. This results in small numerical differences.
    # For "frobenius", there is a bit larger error, possibly because np.linalg.norm relies on BLAS/LAPACK
    # implementations.
    atol = 0
    rtol = 0
    if reduction == "max_mean":
        atol = 1e-6
    if reduction == "frobenius":
        rtol = 1e-3 if is_arm_cpu() else 1e-4

    _check_values(results, atol=atol, rtol=rtol)


@pytest.mark.parametrize("weight_shape", [WEIGHT_SHAPE], ids=[""])
@pytest.mark.parametrize("weight_dtype", SUPPORTED_WEIGHT_DTYPES)
@pytest.mark.parametrize("config", COMPRESSION_CONFIGS, ids=[str(c) for c in COMPRESSION_CONFIGS])
@pytest.mark.parametrize(
    "compression_kwargs",
    [
        {},
        {"awq": True},
        {"scale_estimation": True},
        {"gptq": True},
        {"gptq": True, "scale_estimation": True},
        {"lora_correction": True},
    ],
    ids=["data-free", "awq", "se", "gptq", "gptq_se", "lora"],
)
@pytest.mark.parametrize("dataset_size", [3])
def test_end_to_end_alignment(weight_shape, weight_dtype, config, compression_kwargs, dataset_size):
    def create_ov_model():
        inp = opset.parameter([1, 24, weight_shape[1]])
        weight_const = opset.constant(get_random_float_tensor(weight_shape, weight_dtype, TensorBackend.ov).data)
        weight_const = opset.convert(weight_const, ov.Type.f32)
        matmul = opset.matmul(inp, weight_const, transpose_a=False, transpose_b=True)
        result = opset.result(matmul)
        return ov.Model([result], [inp])

    def create_dataset(model):
        input_data = []
        for i in range(dataset_size):
            input_sample = []
            for j, inp in enumerate(model.inputs):
                partial_shape = inp.get_partial_shape()
                if partial_shape.is_static:
                    input_shape = tuple(inp.shape)
                else:
                    # Batch dimension
                    input_shape = (1 if partial_shape[0].is_dynamic else partial_shape[0].get_length(),)
                    if len(partial_shape) == 2:
                        # Assuming this is sequence length dimension
                        input_shape += (10 if partial_shape[1].is_dynamic else partial_shape[1].get_length(),)
                random_data = get_random_float_tensor(
                    input_shape, TensorDataType.float32, TensorBackend.numpy, seed=hash((i, j)) % (1 << 32)
                ).data
                input_sample.append(random_data)
            input_data.append(input_sample)
        return Dataset(input_data)

    def get_input_node_data(node: ov.Node, input_id: int) -> Tensor:
        # Get the constant node data which is the input to the given node
        child_node = node.input(input_id).get_source_output().get_node()
        if child_node.get_type_name() == "Convert":
            child_node = child_node.input(0).get_source_output().get_node()
        assert child_node.get_type_name() == "Constant"
        return Tensor(get_const_value_as_ov_tensor(child_node)).as_numpy_tensor()

    is_data_aware = (
        compression_kwargs.get("awq")
        or compression_kwargs.get("scale_estimation")
        or compression_kwargs.get("gptq")
        or compression_kwargs.get("lora_correction")
    )

    if is_data_aware and config.mode in [
        CompressWeightsMode.INT8_ASYM,
        CompressWeightsMode.INT8_SYM,
        CompressWeightsMode.MXFP4,
        CompressWeightsMode.FP4,
        CompressWeightsMode.FP8_E4M3,
        CompressWeightsMode.MXFP8_E4M3,
    ]:
        pytest.skip("Data-aware compression is not supported for INT8, MXFP4, FP4, MXFP8, FP8 modes.")
    if config.mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT8_SYM]:
        if weight_dtype in [TensorDataType.f8e4m3, TensorDataType.f8e5m2]:
            pytest.skip("INT8 compression is not supported for f8 dtypes.")
    else:
        compression_kwargs["all_layers"] = True

    results = defaultdict(dict)

    # Iterate over two implementations
    for cb in [ComputationBackend.NumPy, ComputationBackend.OV]:
        # A context manager to enable/disable ov implementation
        with openvino_available(cb == ComputationBackend.OV):
            fn_to_patch = opt_fns.do_integer_quantization if config.is_integer else opt_fns.do_float_quantization
            patch_path = f"nncf.openvino.optimized_functions.{fn_to_patch.__name__}"
            with patch(patch_path, side_effect=fn_to_patch) as mock:
                model = create_ov_model()

                if is_data_aware:
                    compression_kwargs["dataset"] = create_dataset(model)

                nncf.compress_weights(model, mode=config.mode, group_size=config.group_size, **compression_kwargs)

                if cb == ComputationBackend.NumPy:
                    mock.assert_not_called()
                else:
                    mock.assert_called()

                ov_nodes = {node.get_friendly_name(): node for node in model.get_ops()}
                nncf_graph = NNCFGraphFactory.create(model)
                for i, nncf_node in enumerate(nncf_graph.topological_sort()):
                    node_name = nncf_node.node_name
                    node = ov_nodes[node_name]
                    if re.search(r"/fq_weights_\d+$", node_name):
                        # Extract compression-related constants from compression subgraph
                        node_name_prefix = f"{i}_"
                        if "lora" in node_name:
                            node_name_prefix += "lora_A_" if "lora_A" in node_name else "lora_B_"

                        assert node.get_type_name() == "Multiply"
                        mul_node = node
                        results[cb][f"{node_name_prefix}scale"] = get_input_node_data(mul_node, 1)
                        weight_node = node.input(0).get_source_output().get_node()

                        if config.is_asym_mode:
                            assert weight_node.get_type_name() == "Subtract"
                            results[cb][f"{node_name_prefix}zero_point"] = get_input_node_data(weight_node, 1)
                            weight_node = weight_node.input(0).get_source_output().get_node()

                        results[cb][f"{node_name_prefix}weight"] = get_input_node_data(weight_node, 0)

    _check_values(results)


def _get_compression_fn_from_quantization_task(quantization_task, config):
    if quantization_task == QuantizationTask.Q:
        if config.is_integer:
            fn_to_call = do_integer_quantization
            fn_to_patch = opt_fns.do_integer_quantization
        else:
            fn_to_call = do_float_quantization
            fn_to_patch = opt_fns.do_float_quantization
    else:
        if config.is_integer:
            fn_to_call = integer_quantize_dequantize_weight
            fn_to_patch = opt_fns.integer_quantize_dequantize_weight
        else:
            fn_to_call = float_quantize_dequantize_weight
            fn_to_patch = opt_fns.float_quantize_dequantize_weight
    return fn_to_call, fn_to_patch


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
        # zero point must be in ov backend and have (u)int4/nf4/f4e2m1 dtypes in order to be able to insert them into
        # OV model without re-packing
        assert compressed_weight.backend == TensorBackend.ov
        assert compressed_weight.dtype == config.compression_dtype
        if config.is_asym_mode and not precompute_s_zp:
            assert zero_point.backend == TensorBackend.ov
            assert zero_point.dtype == TensorDataType.uint4
    else:
        if quantization_task != QuantizationTask.Q_DQ:
            # Otherwise, for integer compression, compressed weight and zero point must be returned in numpy backend,
            # compressed weight must be of (u)int8, zero point -- in int32; for float compression, the resulting
            # data type and backend depends on the input tensor backend.
            if config.is_integer:
                ref_backend = TensorBackend.numpy
                ref_dtype = TensorDataType.uint8 if config.is_asym_mode else TensorDataType.int8
            else:
                ref_backend = weight_tensor_backend
                ref_dtype = (
                    config.compression_dtype if weight_tensor_backend == TensorBackend.ov else TensorDataType.float32
                )
            assert compressed_weight.backend == ref_backend
            assert compressed_weight.dtype == ref_dtype
            if config.is_asym_mode and not precompute_s_zp:
                assert zero_point.backend == TensorBackend.numpy
                assert zero_point.dtype == TensorDataType.int32
        if quantization_task != QuantizationTask.Q:
            assert decompressed_weight.backend == TensorBackend.numpy
            assert decompressed_weight.dtype == TensorDataType.float32


def _check_values(results, atol=0.0, rtol=0.0):
    def format_list_of_floats(lst, n_first=32):
        return ", ".join(f"{x:.10f}" for x in lst[:n_first])

    # Check that the computed tensors are equal between implementations
    keys = set(results[ComputationBackend.OV]).union(set(results[ComputationBackend.NumPy]))
    for key in keys:
        numpy_result = results[ComputationBackend.NumPy][key]
        ov_result = results[ComputationBackend.OV][key]

        if isinstance(numpy_result, float) and isinstance(ov_result, float):
            numpy_result = Tensor(np.array([numpy_result], dtype=np.float32))
            ov_result = Tensor(np.array([ov_result], dtype=np.float32))

        # Note: For static-shaped OV models doing asymmetric compression with convertable divisions there maybe
        # misalignments equal to 1 quant between OV and NumPy. For more details see ticket 156511.

        try:
            np.testing.assert_allclose(ov_result.data, numpy_result.data, atol=atol, rtol=rtol)
        except AssertionError:
            not_equal_mask = np.not_equal(ov_result.data, numpy_result.data)
            msg = (
                f"Results do not align for {key} with "
                f"{not_equal_mask.sum() / ov_result.data.size * 100:.2f} % misalignment ratio.\n"
                f"OV result (first 32 values):    {format_list_of_floats(ov_result.data[not_equal_mask])}\n"
                f"NumPy result (first 32 values): {format_list_of_floats(numpy_result.data[not_equal_mask])}\n"
            )
            if "input" in results[ComputationBackend.OV] and "input" in results[ComputationBackend.NumPy]:
                numpy_input = results[ComputationBackend.NumPy]["input"].data
                ov_input = results[ComputationBackend.OV]["input"].data
                np.testing.assert_allclose(numpy_input, ov_input, atol=0, rtol=0)
                if "weight" in key:
                    msg += f"Input values (first 32 values)    : {format_list_of_floats(numpy_input[not_equal_mask])}\n"
                misaligned_groups_mask = np.any(not_equal_mask, axis=-1)
                misaligned_groups = numpy_input[misaligned_groups_mask, ...]
                misaligned_groups = np.reshape(misaligned_groups, (-1, misaligned_groups.shape[-1]))
                msg += "First 10 misaligned groups:\n"
                msg += "\n".join(format_list_of_floats(it, misaligned_groups.shape[1]) for it in misaligned_groups[:10])
            raise AssertionError(msg)
