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
import numpy as np
import pytest

from nncf import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.openvino_modeling import OV_MODEL_CACHE
from nncf.quantization.algorithms.weight_compression.openvino_modeling import OVModelParameters
from nncf.quantization.algorithms.weight_compression.openvino_modeling import get_astype_model
from nncf.quantization.algorithms.weight_compression.openvino_modeling import get_compress_decompress_weight_model
from nncf.quantization.algorithms.weight_compression.openvino_modeling import get_compress_weight_model
from nncf.tensor import TensorDataType, Tensor
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
        get_model_fn=get_compress_weight_model,
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
        get_model_fn=get_compress_weight_model,
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
        get_model_fn=get_compress_decompress_weight_model,
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
        get_model_fn=get_compress_decompress_weight_model,
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
    model_getter.get(ov_model_params_kwargs=dict(recompile=recompile))
    ref_size = 0 if recompile else (2 if model_getter._get_model_fn == get_compress_decompress_weight_model else 1)
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
