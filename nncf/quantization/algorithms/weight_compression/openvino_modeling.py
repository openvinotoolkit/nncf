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

import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openvino as ov
from openvino.runtime import opset13 as opset

from nncf.common.utils.decorators import ResultsCacheContainer, cache_results
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.functions.ov import DTYPE_MAP as DTYPE_MAP_OV

TensorList = List[Tensor]
ModelCallable = Callable[[TensorList], TensorList]


OV_MODEL_CACHE = ResultsCacheContainer()


@dataclass
class OVModelParameters:
    input_dtypes: Optional[Dict[str, TensorDataType]] = None
    output_dtypes: Optional[Dict[str, TensorDataType]] = None
    dynamic_shapes: bool = False
    recompile: bool = False
    release_memory: bool = True
    share_inputs: bool = True
    share_outputs: bool = True
    return_ov_tensors: bool = False

    def __hash__(self):
        return hash(
            (
                None if self.output_dtypes is None else frozenset(self.input_dtypes.items()),
                None if self.output_dtypes is None else frozenset(self.output_dtypes.items()),
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
    # Check that input dtypes match the expected dtypes
    for i, inp in enumerate(compiled_model.inputs):
        input_name = inp.any_name
        actual_dtype = inputs[i].dtype
        expected_dtype = ov_model_params.input_dtypes[input_name]
        if actual_dtype != expected_dtype:
            raise ValueError(f"Expected input '{input_name}' to be {expected_dtype}. But found: {actual_dtype}.")

    # Infer the model
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
    return_nodes: Optional[bool] = False,
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
        return_nodes=return_nodes,
        disable_caching=ov_model_params.recompile,
    )


def get_compress_decompress_weight_model(
    ov_model_params: OVModelParameters,
    config: WeightCompressionConfig,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple] = None,
    zero_point_shape: Optional[Tuple] = None,
    reduction_axes: Optional[Tuple] = None,
    return_compressed_weight: Optional[bool] = False,
) -> ModelCallable:
    if ov_model_params.dynamic_shapes:
        weight_shape = (-1,) * len(weight_shape)
        if scale_shape is not None:
            scale_shape = (-1,) * (len(scale_shape) - 1) + (1,)
        if zero_point_shape is not None:
            zero_point_shape = (-1,) * (len(zero_point_shape) - 1) + (1,)

    return _build_compress_decompress_model(
        config,
        ov_model_params,
        weight_shape,
        scale_shape,
        zero_point_shape,
        reduction_axes,
        return_compressed_weight,
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
    input_dtypes = ov_model_params.input_dtypes
    if input_dtypes is None:
        raise ValueError("Input dtypes must be provided.")
    output_dtypes = ov_model_params.output_dtypes
    if output_dtypes is None:
        raise ValueError("Output dtypes must be provided.")

    weight_dtype = input_dtypes.get("weight")
    input_scale_dtype = input_dtypes.get("scale", TensorDataType.float32)
    input_zero_point_dtype = input_dtypes.get("zero_point", TensorDataType.int32)
    compressed_weight_dtype = output_dtypes.get("compressed_weight")
    output_scale_dtype = output_dtypes.get("scale", TensorDataType.float32)
    output_zero_point_dtype = output_dtypes.get("zero_point", TensorDataType.int32)

    # Validate input dtypes
    valid_weight_dtypes = [TensorDataType.float32, TensorDataType.float16, TensorDataType.bfloat16]
    if weight_dtype not in valid_weight_dtypes:
        raise ValueError(
            f"Weight must be one of the following data types: {valid_weight_dtypes}. But found: {weight_dtype}."
        )
    if scale_shape is not None and input_scale_dtype != TensorDataType.float32:
        raise ValueError(f"Input scale must be of float32 data type. But found: {input_scale_dtype}.")
    if zero_point_shape is not None and input_zero_point_dtype != TensorDataType.int32:
        raise ValueError(f"Input zero point must be of int32 data type. But found: {input_zero_point_dtype}.")

    # Validate output dtypes
    valid_compressed_weight_dtypes = [
        TensorDataType.int32,
        TensorDataType.int8,
        TensorDataType.uint8,
        TensorDataType.int4,
        TensorDataType.uint4,
    ]
    if compressed_weight_dtype not in valid_compressed_weight_dtypes + [TensorDataType.float32]:
        raise ValueError(
            f"Compressed weight must be one of the following data types: {valid_compressed_weight_dtypes}. "
            f"But found: {compressed_weight_dtype}."
        )
    if scale_shape is None and output_scale_dtype != TensorDataType.float32:
        raise ValueError(f"Output scale must be of float32 data type. But found: {output_scale_dtype}.")
    is_int_asym = config.is_int_asym
    if is_int_asym and zero_point_shape is None and output_zero_point_dtype not in valid_compressed_weight_dtypes:
        raise ValueError(
            f"Output zero point must be of one of the following data types: {valid_compressed_weight_dtypes}. "
            f"But found: {output_zero_point_dtype}."
        )

    # Build OV model
    weight = opset.parameter(weight_shape, name="weight", dtype=DTYPE_MAP_OV[weight_dtype])
    ov_parameters = [weight]

    num_bits = config.num_bits
    eps = np.finfo(np.float32).eps
    if is_int_asym:
        level_low = 0
        level_high = 2**num_bits - 1
    else:
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1) - 1

    min_values = None
    if scale_shape is not None:
        # Scale is given as an input
        scale = opset.parameter(scale_shape, name="scale", dtype=ov.Type.f32)
        ov_parameters.append(scale)
    else:
        # Compute scale
        if is_int_asym:
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

            scale = opset.select(w_abs_min >= w_max, w_abs_min, opset.negative(w_max))
            scale /= opset.constant(-level_low, ov.Type.f32)
            scale = opset.select(opset.abs(scale) < eps, eps, scale)

    zero_point = None
    if zero_point_shape is not None:
        # Zero point is given as an input
        zero_point = opset.parameter(zero_point_shape, name="zero_point", dtype=ov.Type.i32)
        ov_parameters.append(zero_point)
        # Cast to float32 for an addition later
        zero_point = opset.convert(zero_point, ov.Type.f32)
    elif is_int_asym:
        # Compute zero point
        if min_values is None:
            min_values = opset.reduce_min(
                weight, reduction_axes=reduction_axes, keep_dims=True
            )  # [a1, r, a2] -> [a1, 1, a2]
            min_values = opset.convert(min_values, ov.Type.f32)
        zero_point = opset.constant(level_low, ov.Type.f32) - opset.round(min_values / scale)
        zero_point = opset.clamp(zero_point, level_low, level_high)

    if weight.get_element_type() != ov.Type.f32:
        weight = opset.convert(weight, ov.Type.f32)
    compressed_weight = weight / scale

    if is_int_asym:
        compressed_weight += zero_point

    compressed_weight = opset.round(compressed_weight)
    compressed_weight = opset.clamp(opset.round(compressed_weight), level_low, level_high)
    if compressed_weight_dtype != TensorDataType.float32:
        compressed_weight = opset.convert(compressed_weight, DTYPE_MAP_OV[compressed_weight_dtype])

    ov_results = [compressed_weight]
    if len(ov_parameters) == 1:
        ov_results.append(scale)
        if zero_point is not None:
            zero_point = opset.convert(zero_point, DTYPE_MAP_OV[output_zero_point_dtype])
            ov_results.append(zero_point)

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
    scale_shape: Optional[Tuple] = None,
    zero_point_shape: Optional[Tuple] = None,
    reduction_axes: Optional[Tuple] = None,
    return_compressed_weight: Optional[bool] = False,
) -> ModelCallable:
    input_dtypes = ov_model_params.input_dtypes
    if input_dtypes is None:
        raise ValueError("Input dtypes must be provided.")
    output_dtypes = ov_model_params.output_dtypes
    if output_dtypes is None:
        raise ValueError("Output dtypes must be provided.")

    decompressed_weight_dtype = output_dtypes.get("decompressed_weight")
    if decompressed_weight_dtype != TensorDataType.float32:
        raise ValueError(f"Decompressed weight must be of float32 data type. But found: {decompressed_weight_dtype}.")

    if "compressed_weight" not in output_dtypes:
        ov_model_params = copy.deepcopy(ov_model_params)
        ov_model_params.output_dtypes["compressed_weight"] = TensorDataType.float32

    ov_parameters, ov_results = get_compress_weight_model(
        ov_model_params, config, weight_shape, scale_shape, zero_point_shape, reduction_axes, return_nodes=True
    )

    if config.is_int_asym:
        if len(ov_parameters) == 1:
            # weight -> compressed_weight, scale, zero_point
            compressed_weight, scale, zero_point = ov_results
        else:
            # weight, scale, zero_point -> compressed_weight
            compressed_weight = ov_results[0]
            scale, zero_point = ov_parameters[1:]

        compressed_weight = opset.convert(compressed_weight, ov.Type.i32) - opset.convert(zero_point, ov.Type.i32)
    else:
        if len(ov_parameters) == 1:
            # weight -> compressed_weight, scale
            compressed_weight, scale = ov_results
        else:
            # weight, scale -> compressed_weight
            compressed_weight = ov_results[0]
            scale = ov_parameters[1]

    if compressed_weight.get_element_type() != ov.Type.f32:
        compressed_weight = opset.convert(compressed_weight, ov.Type.f32)
    decompressed_weight = opset.multiply(scale, compressed_weight)

    ov_results = [decompressed_weight] + ov_results if return_compressed_weight else [decompressed_weight]
    model = ov.Model(ov_results, ov_parameters)
    compiled_model = ov.compile_model(model, device_name="CPU")

    return partial(run_model, ov_model_params, compiled_model, ov_model_params.return_ov_tensors)


def get_astype_model(ov_model_params: OVModelParameters, input_shape: Tuple) -> ModelCallable:
    if ov_model_params.dynamic_shapes:
        input_shape = (-1,) * len(input_shape)
    return _build_astype_model(ov_model_params, input_shape, disable_caching=ov_model_params.recompile)


@cache_results(OV_MODEL_CACHE)
def _build_astype_model(ov_model_params: OVModelParameters, arg_shape: Tuple) -> ModelCallable:
    input_dtypes = ov_model_params.input_dtypes
    if input_dtypes is None:
        raise ValueError("Input dtypes must be provided.")
    output_dtypes = ov_model_params.output_dtypes
    if output_dtypes is None:
        raise ValueError("Output dtypes must be provided.")
    if "input" not in input_dtypes:
        raise ValueError("Input dtype is required.")
    if "output" not in output_dtypes:
        raise ValueError("Output dtype is required.")

    arg = opset.parameter(arg_shape, dtype=DTYPE_MAP_OV[input_dtypes["input"]], name="input")
    res = opset.convert(arg, DTYPE_MAP_OV[output_dtypes["output"]])
    model = ov.Model([res], [arg])
    compiled_model = ov.compile_model(model, device_name="CPU")

    return partial(run_model, ov_model_params, compiled_model, ov_model_params.return_ov_tensors)
