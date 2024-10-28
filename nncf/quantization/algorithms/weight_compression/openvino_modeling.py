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
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import openvino as ov
from openvino.runtime import opset13 as opset

from nncf import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.results_caching import ResultsCacheContainer
from nncf.results_caching import cache_results
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.functions.ov import DTYPE_MAP as OV_DTYPE_MAP

TensorList = List[Tensor]
ModelCallable = Callable[[TensorList], TensorList]


OV_MODEL_CACHE = ResultsCacheContainer()


@dataclass
class OVModelParameters:
    input_dtype: TensorDataType
    output_dtype: Optional[TensorDataType] = None
    dynamic_shapes: bool = False
    recompile: bool = False
    release_memory: bool = True
    share_inputs: bool = True
    share_outputs: bool = True
    return_ov_tensors: bool = False

    def __hash__(self):
        return hash(
            (
                self.input_dtype,
                self.dynamic_shapes,
                self.recompile,
                self.release_memory,
                self.share_inputs,
                self.share_outputs,
                self.return_ov_tensors,
            )
        )


def run_model(
    ov_model_params: OVModelParameters, compiled_model: ov.CompiledModel, return_ov_tensors: bool, inputs: TensorList
) -> TensorList:
    if any(isinstance(it, Tensor) for it in inputs):
        inputs = [inp.data for inp in inputs]

    if return_ov_tensors:
        infer_request = compiled_model.create_infer_request()
        infer_request.infer(
            inputs, share_inputs=ov_model_params.share_inputs, share_outputs=ov_model_params.share_outputs
        )
        outputs = [infer_request.get_output_tensor(i) for i in range(len(infer_request.results))]
    else:
        outputs = compiled_model(
            inputs, share_inputs=ov_model_params.share_inputs, share_outputs=ov_model_params.share_outputs
        )
        outputs = [outputs[i] for i in range(len(outputs))]
    outputs = [Tensor(it) for it in outputs]

    if ov_model_params.release_memory:
        compiled_model.release_memory()

    return outputs


def get_compress_weight_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple] = None,
    zero_point_shape: Optional[Tuple] = None,
    reduction_axes: Optional[Tuple] = None,
) -> ModelCallable:
    if scale_shape is None and zero_point_shape is not None:
        raise Exception("Zero point shape can only be provided if scale shape is provided.")

    if ov_model_params.dynamic_shapes:
        weight_shape = (-1,) * len(weight_shape)
        if scale_shape is not None:
            scale_shape = (-1,) * (len(scale_shape) - 1) + (1,)
        if zero_point_shape is not None:
            zero_point_shape = (-1,) * (len(zero_point_shape) - 1) + (1,)

    return _build_compress_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        zero_point_shape,
        reduction_axes,
        return_nodes=False,
        disable_caching=ov_model_params.recompile,
    )


def get_compress_decompress_weight_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple],
    zero_point_shape: Optional[Tuple] = None,
) -> ModelCallable:
    if ov_model_params.dynamic_shapes:
        weight_shape = (-1,) * len(weight_shape)
        scale_shape = (-1,) * (len(scale_shape) - 1) + (1,)
        if zero_point_shape is not None:
            zero_point_shape = (-1,) * (len(zero_point_shape) - 1) + (1,)

    return _build_compress_decompress_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        zero_point_shape,
        disable_caching=ov_model_params.recompile,
    )


@cache_results(OV_MODEL_CACHE)
def _build_compress_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple] = None,
    zero_point_shape: Optional[Tuple] = None,
    reduction_axes: Optional[Tuple] = None,
    return_nodes: bool = False,
) -> Union[ModelCallable, Tuple[List[ov._pyopenvino.Node], List[ov._pyopenvino.Node]]]:
    weight = opset.parameter(weight_shape, name="w", dtype=OV_DTYPE_MAP[ov_model_params.input_dtype])
    ov_parameters = [weight]

    num_bits = config.num_bits
    eps = np.finfo(np.float32).eps
    if config.is_int_asym:
        level_low = 0
        level_high = 2**num_bits - 1
    else:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1) - 1

    min_values = None
    if scale_shape is not None:
        # Scale is given as an input
        scale = opset.parameter(scale_shape, name="s", dtype=ov.Type.f32)
        ov_parameters.append(scale)
    else:
        # Compute scale
        if config.is_int_asym:
            min_values = opset.reduce_min(
                weight, reduction_axes=reduction_axes, keep_dims=True
            )  # [a1, r, a2] -> [a1, 1, a2]
            max_values = opset.reduce_max(
                weight, reduction_axes=reduction_axes, keep_dims=True
            )  # [a1, r, a2] -> [a1, 1, a2]
            min_values, max_values = opset.convert(min_values, ov.Type.f32), opset.convert(max_values, ov.Type.f32)

            levels = level_high - level_low + 1
            scale = (max_values - min_values) / opset.constant(levels - 1, ov.Type.f32)
            scale = opset.select(opset.abs(scale) < eps, eps, scale)
        else:
            w_abs_min = opset.abs(opset.reduce_min(weight, reduction_axes=reduction_axes, keep_dims=True))
            w_max = opset.reduce_max(weight, reduction_axes=reduction_axes, keep_dims=True)
            w_abs_min, w_max = opset.convert(w_abs_min, ov.Type.f32), opset.convert(w_max, ov.Type.f32)

            scale = opset.select(w_abs_min >= w_max, w_abs_min, opset.constant(0, ov.Type.f32) - w_max)
            scale /= opset.constant(level_high, ov.Type.f32)
            scale = opset.select(opset.abs(scale) < eps, eps, scale)

    zero_point = None
    if zero_point_shape is not None:
        # Zero point is given as an input
        zero_point = opset.parameter(zero_point_shape, name="zp", dtype=ov.Type.i32)
        ov_parameters.append(zero_point)
        zero_point = opset.convert(zero_point, ov.Type.f32)
    elif config.is_int_asym:
        # Compute zero point
        if min_values is None:
            min_values = opset.reduce_min(
                weight, reduction_axes=reduction_axes, keep_dims=True
            )  # [a1, r, a2] -> [a1, 1, a2]
            min_values = opset.convert(min_values, ov.Type.f32)

        level_low = 0
        level_high = 2**num_bits - 1
        zero_point = opset.constant(level_low, ov.Type.f32) - opset.round(min_values / scale)
        zero_point = opset.clamp(zero_point, level_low, level_high)

    if weight.get_element_type() != ov.Type.f32:
        weight = opset.convert(weight, ov.Type.f32)
    compressed_w = weight / scale

    if config.is_int_asym:
        if ov_model_params.output_dtype is not None:
            dtype = OV_DTYPE_MAP[ov_model_params.output_dtype]
        else:
            dtype = ov.Type.u8 if config.mode == CompressWeightsMode.INT8_ASYM else ov.Type.u4
        compressed_w += zero_point
    else:
        if ov_model_params.output_dtype is not None:
            dtype = OV_DTYPE_MAP[ov_model_params.output_dtype]
        else:
            dtype = ov.Type.i8 if config.mode == CompressWeightsMode.INT8_SYM else ov.Type.i4

    compressed_w = opset.clamp(opset.round(compressed_w), level_low, level_high)
    compressed_w = opset.convert(compressed_w, dtype, name="compressed_weights")

    ov_results = [compressed_w]
    if len(ov_parameters) != 3:
        # Two cases:
        #   1. weight -> compressed_weight, scale, (zero_point)
        #   2. weight, scale -> compressed_weight, (zero_point)
        if len(ov_parameters) == 1:
            ov_results.append(scale)

        if zero_point is not None:
            ov_results.append(opset.convert(zero_point, compressed_w.get_element_type()))

    if return_nodes:
        return ov_parameters, ov_results

    model = ov.Model(ov_results, ov_parameters)
    compiled_model = ov.compile_model(model, device_name="CPU")

    return partial(run_model, ov_model_params, compiled_model, ov_model_params.return_ov_tensors)


@cache_results(OV_MODEL_CACHE)
def _build_compress_decompress_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: Tuple,
    scale_shape: Tuple,
    zero_point_shape: Optional[Tuple] = None,
) -> ModelCallable:
    ov_parameters, ov_results = _build_compress_model(
        config, ov_model_params, weight_shape, scale_shape, zero_point_shape, reduction_axes=None, return_nodes=True
    )

    if config.is_int_asym:
        if len(ov_parameters) == 1:
            # weight -> compressed_weight, scale, zero_point
            compressed_w, scale, zero_point = ov_results
        elif len(ov_parameters) == 2:
            # weight, scale -> compressed_weight, zero_point
            compressed_w, zero_point = ov_results
            scale = ov_parameters[1]
        else:
            # weight, scale, zero_point -> compressed_weight
            compressed_w = ov_results[0]
            scale, zero_point = ov_parameters[1:]
        decompressed_w = opset.convert(opset.convert(compressed_w, ov.Type.i32) - zero_point, ov.Type.f32) * scale
    else:
        if len(ov_parameters) == 1:
            # weight -> compressed_weight, scale
            compressed_w, scale = ov_results
        else:
            # weight, scale -> compressed_weight
            compressed_w = ov_results[0]
            scale = ov_parameters[1]
        decompressed_w = opset.convert(compressed_w, ov.Type.f32) * scale

    model = ov.Model([decompressed_w], ov_parameters)
    compiled_model = ov.compile_model(model, device_name="CPU")

    return partial(run_model, ov_model_params, compiled_model, ov_model_params.return_ov_tensors)


def get_astype_model(ov_model_params: OVModelParameters, arg_shape: Tuple, dtype: TensorDataType) -> ModelCallable:
    if ov_model_params.dynamic_shapes:
        arg_shape = (-1,) * len(arg_shape)
    return _build_astype_model(ov_model_params, arg_shape, dtype)


@cache_results(OV_MODEL_CACHE)
def _build_astype_model(ov_model_params: OVModelParameters, arg_shape: Tuple, dtype: TensorDataType) -> ModelCallable:
    arg = opset.parameter(arg_shape, dtype=OV_DTYPE_MAP[ov_model_params.input_dtype])
    res = opset.convert(arg, OV_DTYPE_MAP[dtype])
    model = ov.Model([res], [arg])
    compiled_model = ov.compile_model(model, device_name="CPU")

    return partial(run_model, ov_model_params, compiled_model, ov_model_params.return_ov_tensors)
