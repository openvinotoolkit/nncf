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
import numpy as np
import pytest

from nncf import CompressWeightsMode
from nncf.common.utils.caching import disable_results_caching
from nncf.openvino.cpu_info import is_arm_cpu
from nncf.openvino.optimized_functions.models import OV_MODEL_CACHE
from nncf.openvino.optimized_functions.models import OVModelParameters
from nncf.openvino.optimized_functions.models import _infer_ov_model
from nncf.openvino.optimized_functions.models import get_astype_model
from nncf.openvino.optimized_functions.models import get_float_quantization_model
from nncf.openvino.optimized_functions.models import get_float_quantize_dequantize_weight_model
from nncf.openvino.optimized_functions.models import get_integer_quantization_error_model
from nncf.openvino.optimized_functions.models import get_integer_quantization_model
from nncf.openvino.optimized_functions.models import get_integer_quantize_dequantize_weight_model
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP as DTYPE_MAP_NP


class ModelGetter:
    def __init__(self, get_model_fn, ov_model_params_kwargs, get_model_kwargs):
        self._get_model_fn = get_model_fn
        self._ov_model_params_kwargs = ov_model_params_kwargs
        self._get_model_kwargs = get_model_kwargs

    def get(self, ov_model_params_kwargs=None, get_model_kwargs=None):
        ov_model_params_kwargs = ov_model_params_kwargs or {}
        get_model_kwargs = get_model_kwargs or {}
        return self._get_model_fn(
            OVModelParameters(**{**self._ov_model_params_kwargs, **ov_model_params_kwargs}),
            **{**self._get_model_kwargs, **get_model_kwargs},
        )


MODEL_GETTERS = [
    ModelGetter(
        get_model_fn=get_integer_quantization_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
                "scale": TensorDataType.float32,
                "zero_point": TensorDataType.int32,
            },
            output_dtypes={"compressed_weight": TensorDataType.uint8},
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.INT8_ASYM),
            weight_shape=(10, 4),
            scale_shape=(10, 1),
            zero_point_shape=(10, 1),
        ),
    ),
    ModelGetter(
        get_model_fn=get_integer_quantization_model,
        ov_model_params_kwargs=dict(
            input_dtypes={"weight": TensorDataType.float32},
            output_dtypes={
                "compressed_weight": TensorDataType.uint8,
                "scale": TensorDataType.float32,
                "zero_point": TensorDataType.int32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.INT8_ASYM),
            weight_shape=(10, 4),
            reduction_axes=(1,),
        ),
    ),
    ModelGetter(
        get_model_fn=get_integer_quantize_dequantize_weight_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
                "scale": TensorDataType.float32,
                "zero_point": TensorDataType.int32,
            },
            output_dtypes={
                "decompressed_weight": TensorDataType.float32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.INT8_ASYM),
            weight_shape=(10, 4),
            scale_shape=(10, 1),
            zero_point_shape=(10, 1),
        ),
    ),
    ModelGetter(
        get_model_fn=get_integer_quantize_dequantize_weight_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
            },
            output_dtypes={
                "decompressed_weight": TensorDataType.float32,
                "compressed_weight": TensorDataType.int32,
                "scale": TensorDataType.float32,
                "zero_point": TensorDataType.int32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.INT8_ASYM),
            weight_shape=(10, 4),
            reduction_axes=(1,),
            return_compressed_weight=True,
        ),
    ),
    ModelGetter(
        get_model_fn=get_astype_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "input": TensorDataType.float32,
            },
            output_dtypes={
                "output": TensorDataType.bfloat16,
            },
        ),
        get_model_kwargs=dict(
            input_shape=(10, 4),
        ),
    ),
    ModelGetter(
        get_model_fn=get_integer_quantization_error_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.INT4_ASYM, group_size=2),
            original_weight_shape=(10, 4),
            weight_shape=(10, 2, 2),
            original_reduction_axes=(1,),
            reduction_axes=(2,),
            reduction="max_mean",
        ),
    ),
    ModelGetter(
        get_model_fn=get_float_quantization_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
                "scale": TensorDataType.float32,
            },
            output_dtypes={"compressed_weight": TensorDataType.float32},
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.NF4),
            weight_shape=(10, 4),
            scale_shape=(10, 1),
        ),
    ),
    ModelGetter(
        get_model_fn=get_float_quantization_model,
        ov_model_params_kwargs=dict(
            input_dtypes={"weight": TensorDataType.float32},
            output_dtypes={
                "compressed_weight": TensorDataType.float32,
                "scale": TensorDataType.float32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.NF4),
            weight_shape=(10, 4),
            reduction_axes=(1,),
        ),
    ),
    ModelGetter(
        get_model_fn=get_float_quantize_dequantize_weight_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
                "scale": TensorDataType.float32,
            },
            output_dtypes={
                "decompressed_weight": TensorDataType.float32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.NF4),
            weight_shape=(10, 4),
            scale_shape=(10, 1),
        ),
    ),
    ModelGetter(
        get_model_fn=get_float_quantize_dequantize_weight_model,
        ov_model_params_kwargs=dict(
            input_dtypes={
                "weight": TensorDataType.float32,
            },
            output_dtypes={
                "decompressed_weight": TensorDataType.float32,
                "compressed_weight": TensorDataType.float32,
                "scale": TensorDataType.float32,
            },
        ),
        get_model_kwargs=dict(
            config=WeightCompressionConfig(CompressWeightsMode.NF4),
            weight_shape=(10, 4),
            reduction_axes=(1,),
            return_compressed_weight=True,
        ),
    ),
]


@pytest.mark.parametrize(
    "model_getter,input_shapes,ref_cache_size",
    [
        (
            MODEL_GETTERS[0],
            [
                dict(weight_shape=(10, 4), scale_shape=(10, 1), zero_point_shape=(10, 1)),
                dict(weight_shape=(20, 6), scale_shape=(20, 1), zero_point_shape=(20, 1)),
                dict(weight_shape=(20, 8), scale_shape=(20, 1), zero_point_shape=(20, 1)),
                dict(weight_shape=(10, 4, 4), scale_shape=(10, 4, 1), zero_point_shape=(10, 4, 1)),
                dict(weight_shape=(10, 8, 4), scale_shape=(10, 8, 1), zero_point_shape=(10, 8, 1)),
            ],
            {False: 5, True: 2},
        ),
        (
            MODEL_GETTERS[1],
            [
                dict(weight_shape=(10, 4)),
                dict(weight_shape=(20, 6)),
                dict(weight_shape=(20, 8)),
                dict(weight_shape=(10, 4, 4)),
                dict(weight_shape=(10, 8, 4)),
            ],
            {False: 5, True: 2},
        ),
        (
            MODEL_GETTERS[2],
            [
                dict(weight_shape=(10, 4), scale_shape=(10, 1), zero_point_shape=(10, 1)),
                dict(weight_shape=(20, 6), scale_shape=(20, 1), zero_point_shape=(20, 1)),
                dict(weight_shape=(20, 8), scale_shape=(20, 1), zero_point_shape=(20, 1)),
                dict(weight_shape=(10, 4, 4), scale_shape=(10, 4, 1), zero_point_shape=(10, 4, 1)),
                dict(weight_shape=(10, 8, 4), scale_shape=(10, 8, 1), zero_point_shape=(10, 8, 1)),
            ],
            {False: 10, True: 4},
        ),
        (
            MODEL_GETTERS[3],
            [
                dict(weight_shape=(10, 4)),
                dict(weight_shape=(20, 6)),
                dict(weight_shape=(20, 8)),
                dict(weight_shape=(10, 4, 4)),
                dict(weight_shape=(10, 8, 4)),
            ],
            {False: 10, True: 4},
        ),
        (
            MODEL_GETTERS[4],
            [
                dict(input_shape=(10, 1)),
                dict(input_shape=(10, 2)),
                dict(input_shape=(20, 3)),
                dict(input_shape=(10, 4, 4)),
                dict(input_shape=(10, 8, 4)),
            ],
            {False: 5, True: 2},
        ),
        (
            MODEL_GETTERS[6],
            [
                dict(weight_shape=(10, 4), scale_shape=(10, 1)),
                dict(weight_shape=(20, 6), scale_shape=(20, 1)),
                dict(weight_shape=(20, 8), scale_shape=(20, 1)),
                dict(weight_shape=(10, 4, 4), scale_shape=(10, 4, 1)),
                dict(weight_shape=(10, 8, 4), scale_shape=(10, 8, 1)),
            ],
            {False: 5, True: 2},
        ),
        (
            MODEL_GETTERS[7],
            [
                dict(weight_shape=(10, 4)),
                dict(weight_shape=(20, 6)),
                dict(weight_shape=(20, 8)),
                dict(weight_shape=(10, 4, 4)),
                dict(weight_shape=(10, 8, 4)),
            ],
            {False: 5, True: 2},
        ),
        (
            MODEL_GETTERS[8],
            [
                dict(weight_shape=(10, 4), scale_shape=(10, 1)),
                dict(weight_shape=(20, 6), scale_shape=(20, 1)),
                dict(weight_shape=(20, 8), scale_shape=(20, 1)),
                dict(weight_shape=(10, 4, 4), scale_shape=(10, 4, 1)),
                dict(weight_shape=(10, 8, 4), scale_shape=(10, 8, 1)),
            ],
            {False: 10, True: 4},
        ),
        (
            MODEL_GETTERS[9],
            [
                dict(weight_shape=(10, 4)),
                dict(weight_shape=(20, 6)),
                dict(weight_shape=(20, 8)),
                dict(weight_shape=(10, 4, 4)),
                dict(weight_shape=(10, 8, 4)),
            ],
            {False: 10, True: 4},
        ),
    ],
)
@pytest.mark.parametrize("dynamic_shapes", [False, True])
def test_dynamic_shapes(model_getter, input_shapes, ref_cache_size, dynamic_shapes):
    # Check that model cache contains fewer elements with dynamic shapes enabled
    OV_MODEL_CACHE.clear()
    for shape_kwargs in input_shapes:
        model_getter.get(ov_model_params_kwargs=dict(dynamic_shapes=dynamic_shapes), get_model_kwargs=shape_kwargs)
    assert len(OV_MODEL_CACHE._cache) == ref_cache_size[dynamic_shapes]


@pytest.mark.parametrize("model_getter", MODEL_GETTERS)
@pytest.mark.parametrize("recompile", [True, False])
def test_recompile(model_getter, recompile):
    # Check that with recompilation ov models are not cached
    OV_MODEL_CACHE.clear()
    if recompile:
        with disable_results_caching(OV_MODEL_CACHE):
            model_getter.get()
    else:
        model_getter.get()
    if recompile:
        ref_size = 0
    elif model_getter._get_model_fn in [
        get_integer_quantize_dequantize_weight_model,
        get_float_quantize_dequantize_weight_model,
    ]:
        ref_size = 2
    elif model_getter._get_model_fn == get_integer_quantization_error_model:
        ref_size = 3
    else:
        ref_size = 1

    assert len(OV_MODEL_CACHE._cache) == ref_size



@pytest.mark.parametrize("model_getter", MODEL_GETTERS)
@pytest.mark.parametrize("return_ov_tensors", [True, False])
def test_return_ov_tensors(model_getter, return_ov_tensors):
    # Check that ov tensors are returned
    OV_MODEL_CACHE.clear()
    inputs = []
    for input_name, input_dtype in model_getter._ov_model_params_kwargs["input_dtypes"].items():
        input_shape = model_getter._get_model_kwargs.get(f"{input_name}_shape")
        inputs.append(Tensor(np.zeros(input_shape, dtype=DTYPE_MAP_NP[input_dtype])))

    model_run_fn = model_getter.get(ov_model_params_kwargs=dict(return_ov_tensors=return_ov_tensors))
    outputs = model_run_fn(inputs)

    assert all([out.backend == (TensorBackend.ov if return_ov_tensors else TensorBackend.numpy) for out in outputs])


@pytest.mark.parametrize("release_memory", [True, False])
def test_release_memory(mocker, release_memory):
    compiled_model = mocker.Mock()
    compiled_model.release_memory = mocker.Mock()

    input_mock = mocker.Mock()
    input_mock.any_name = "input"
    compiled_model.inputs = [input_mock]

    output_mock = mocker.Mock()
    infer_request = mocker.Mock()
    infer_request.infer.return_value = [output_mock]
    compiled_model._infer_request = infer_request

    ov_model_params = OVModelParameters(input_dtypes={"input": TensorDataType.float32}, release_memory=release_memory)
    input_tensor = mocker.Mock()
    input_tensor.dtype = TensorDataType.float32
    input_tensor.data = [1, 2, 3]
    inputs = [input_tensor]

    _infer_ov_model(ov_model_params, compiled_model, inputs=inputs)
    if release_memory:
        compiled_model.release_memory.assert_called_once()
    else:
        compiled_model.release_memory.assert_not_called()


@pytest.mark.parametrize("share_inputs", [True, False])
@pytest.mark.parametrize("share_outputs", [True, False])
@pytest.mark.parametrize("return_ov_tensors", [True, False])
def test_share_inputs_outputs(mocker, share_inputs, share_outputs, return_ov_tensors):
    compiled_model = mocker.Mock()

    input_mock = mocker.Mock()
    input_mock.any_name = "input"
    compiled_model.inputs = [input_mock]

    output_mock = mocker.Mock()

    infer_request = mocker.Mock()
    infer_request.infer.return_value = [output_mock]
    if return_ov_tensors:
        infer_request.get_output_tensor.return_value = output_mock
    compiled_model._infer_request = infer_request

    ov_model_params = OVModelParameters(
        input_dtypes={"input": TensorDataType.float32},
        return_ov_tensors=return_ov_tensors,
        share_inputs=share_inputs,
        share_outputs=share_outputs,
    )

    input_tensor = mocker.Mock()
    input_tensor.dtype = TensorDataType.float32
    input_tensor.data = [1, 2, 3]
    inputs = [input_tensor]

    _infer_ov_model(ov_model_params, compiled_model, inputs=inputs)

    infer_request.infer.assert_called_once_with(
        [input_tensor.data], share_inputs=share_inputs, share_outputs=share_outputs
    )


@pytest.mark.parametrize(
    "weight,convertable_division,ref_compressed_weight",
    [
        ([[0.70361328125, 0.92919921875, 0.37109375, -0.98974609375]], True, [[225, 255, 181, 0]]),
        ([[0.70361328125, 0.92919921875, 0.37109375, -0.98974609375]], False, [[226, 255, 181, 0]]),
    ],
)
def test_convertable_divison(weight, convertable_division, ref_compressed_weight):
    ov_model_params = OVModelParameters(
        input_dtypes={"weight": TensorDataType.float32},
        dynamic_shapes=not convertable_division,
        convertable_division=convertable_division,
    )
    config = WeightCompressionConfig(CompressWeightsMode.INT8_ASYM)

    weight = np.array(weight, np.float32)
    ref_compressed_weight = np.array(ref_compressed_weight, np.uint8)
    model_run_fn = get_integer_quantization_model(ov_model_params, config, weight.shape, reduction_axes=(1,))
    compressed_weight = model_run_fn([Tensor(weight)])[0]
    np.testing.assert_allclose(compressed_weight.data, ref_compressed_weight, atol=0, rtol=0)
