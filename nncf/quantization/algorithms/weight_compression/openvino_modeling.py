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
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import openvino as ov
from openvino.runtime import opset13 as opset

import nncf
from nncf import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig


@dataclass
class OVModelParameters:
    dynamic: bool = False
    recompile: bool = False
    release_memory: bool = True
    share_outputs: bool = True
    input_dtype: str = "fp32"

    def __hash__(self):
        return hash((self.dynamic, self.recompile, self.release_memory, self.share_outputs, self.input_dtype))


class CompiledModelCache:
    def __init__(self):
        self._cache = {}

    def clear(self):
        self._cache.clear()

    def is_empty(self):
        return len(self._cache) == 0


COMPILED_MODEL_CACHE = CompiledModelCache()


def clear_cache():
    COMPILED_MODEL_CACHE.clear()


def cache_results(func):
    def wrapper(*args, **kwargs):
        sig = inspect.signature(func)
        new_kwargs = {name: arg for name, arg in zip(sig.parameters, args)}
        new_kwargs.update(kwargs)
        cache_key = (func.__name__, frozenset(new_kwargs.items()))
        recompile = new_kwargs.get("ov_model_params", OVModelParameters()).recompile
        cache = COMPILED_MODEL_CACHE._cache
        if not recompile and cache_key in cache:
            return cache[cache_key]
        result = func(*args, **kwargs)
        cache[cache_key] = result
        return result

    return wrapper


@cache_results
def get_compress_weight_model(
    config: WeightCompressionConfig,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple] = None,
    zero_point_shape: Optional[Tuple] = None,
    reduction_axes: Optional[Tuple] = None,
    ov_model_params: Optional[OVModelParameters] = None,
):
    if scale_shape is None and zero_point_shape is not None:
        raise Exception("Zero point shape can only be provided if scale shape is provided.")
    # if (scale_shape is None) != (reduction_axes is not None):
    #     raise Exception("Either one of scale_shape or reduction_axes must be provided at the same time.")

    if ov_model_params.dynamic:
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
    )


@cache_results
def get_compress_decompress_weight_model(
    config: WeightCompressionConfig,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple],
    zero_point_shape: Optional[Tuple] = None,
    ov_model_params: Optional[OVModelParameters] = None,
):
    if ov_model_params is None:
        ov_model_params = OVModelParameters()
    if config.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT4_SYM]:
        ov_model_params.dynamic = False

    if ov_model_params.dynamic:
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
    )


def _build_compress_decompress_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: Tuple,
    scale_shape: Tuple,
    zero_point_shape: Optional[Tuple] = None,
):
    ov_parameters, ov_results = _build_compress_model(
        config, ov_model_params, weight_shape, scale_shape, zero_point_shape, reduction_axes=None, return_nodes=True
    )
    return _get_compress_decompress_model(
        config,
        ov_model_params,
        ov_parameters,
        ov_results,
    )


def _build_compress_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    weight_shape: Tuple,
    scale_shape: Optional[Tuple] = None,
    zero_point_shape: Optional[Tuple] = None,
    reduction_axes: Optional[Tuple] = None,
    return_nodes: bool = False,
):
    if ov_model_params.input_dtype == "fp32":
        input_dtype = ov.Type.f32
    elif ov_model_params.input_dtype == "fp16":
        input_dtype = ov.Type.f16
    elif ov_model_params.input_dtype == "bf16":
        input_dtype = ov.Type.bf16
    else:
        raise Exception
    weight = opset.parameter(weight_shape, name="w", dtype=input_dtype)
    ov_parameters = [weight]

    if scale_shape is not None:
        # Compute only the compressed weight

        scale = opset.parameter(scale_shape, name="s", dtype=ov.Type.f32)
        ov_parameters.append(scale)

        zero_point = None
        if config.mode in [CompressWeightsMode.INT8_ASYM, config.mode.INT4_ASYM]:
            zero_point = opset.parameter(zero_point_shape, name="zp", dtype=ov.Type.i32)
            ov_parameters.append(zero_point)
    else:
        # Compute compressed weight, scale and, possibly, zero point

        group_size = config.group_size
        if group_size != -1:
            if isinstance(reduction_axes, tuple) and len(reduction_axes) == 1:
                reduction_axes = reduction_axes[0]
            if not isinstance(reduction_axes, int):
                raise NotImplementedError(
                    f"Group-wise quantization expects a single reduction axis, but given: {reduction_axes}."
                )
            channel_size = weight.shape[reduction_axes]
            if channel_size % group_size != 0:
                raise nncf.ValidationError(
                    f"Channel size {channel_size} should be divisible by size of group {group_size}"
                )

            num_groups_per_channel = channel_size // group_size
            shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
            shape[reduction_axes : reduction_axes + 1] = (num_groups_per_channel, group_size)
            weight = opset.reshape(weight, shape, special_zero=False)
            reduction_axes += 1

        mode = config.mode
        num_bits = config.num_bits
        eps = np.finfo(np.float32).eps
        if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]:
            min_values = opset.reduce_min(
                weight, reduction_axes=reduction_axes, keep_dims=True
            )  # [a1, r, a2] -> [a1, 1, a2]
            max_values = opset.reduce_max(
                weight, reduction_axes=reduction_axes, keep_dims=True
            )  # [a1, r, a2] -> [a1, 1, a2]
            min_values, max_values = opset.convert(min_values, ov.Type.f32), opset.convert(max_values, ov.Type.f32)

            level_low = 0
            level_high = 2**num_bits - 1
            levels = level_high - level_low + 1
            scale = (max_values - min_values) / opset.constant(levels - 1, ov.Type.f32)
            scale = opset.select(opset.abs(scale) < eps, eps, scale)

            zero_point = opset.constant(level_low, ov.Type.f32) - opset.round(min_values / scale)
            zero_point = opset.clamp(zero_point, level_low, level_high)
        else:
            zero_point = None
            level_high = opset.constant(2 ** (num_bits - 1), ov.Type.f32)

            w_abs_min = opset.abs(opset.reduce_min(weight, reduction_axes=reduction_axes, keep_dims=True))
            w_max = opset.reduce_max(weight, reduction_axes=reduction_axes, keep_dims=True)
            w_abs_min, w_max = opset.convert(w_abs_min, ov.Type.f32), opset.convert(w_max, ov.Type.f32)

            scale = opset.select(w_abs_min >= w_max, w_abs_min, opset.constant(0, ov.Type.f32) - w_max)
            scale /= level_high
            scale = opset.select(opset.abs(scale) < eps, eps, scale)

    return _get_compress_model(
        config,
        ov_model_params,
        ov_parameters,
        weight,
        scale,
        zero_point,
        return_nodes,
    )


def _get_compress_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    ov_parameters: List[ov._pyopenvino.op.Parameter],
    w: ov.runtime.Node,
    s: ov.runtime.Node,
    zp: Optional[ov.runtime.Node] = None,
    return_nodes: Optional[bool] = False,
):
    if w.get_element_type() != ov.Type.f32:
        w = opset.convert(w, ov.Type.f32)

    compressed_w = w / s

    num_bits = config.num_bits
    if config.mode in [CompressWeightsMode.INT8_ASYM, config.mode.INT4_ASYM]:
        # dtype = ov.Type.u8
        dtype = ov.Type.u8 if config.mode == CompressWeightsMode.INT8_ASYM else ov.Type.u4
        level_low = 0
        level_high = 2**num_bits - 1
        compressed_w += opset.convert(zp, ov.Type.f32)
    elif config.mode in [CompressWeightsMode.INT8_SYM, config.mode.INT4_SYM]:
        # dtype = ov.Type.i8
        dtype = ov.Type.i8 if config.mode == CompressWeightsMode.INT8_SYM else ov.Type.u4
        level_low = -(2 ** (num_bits - 1))
        level_high = 2 ** (num_bits - 1) - 1
    else:
        raise Exception

    compressed_w = opset.clamp(opset.round(compressed_w), level_low, level_high)
    compressed_w = opset.convert(compressed_w, dtype, name="compressed_weights")

    ov_results = [compressed_w]
    if len(ov_parameters) == 1:
        ov_results.append(s)
        if zp is not None:
            ov_results.append(opset.convert(zp, compressed_w.get_element_type()))

    if return_nodes:
        return ov_parameters, ov_results

    model = ov.Model(ov_results, ov_parameters)
    compiled_model = ov.compile_model(model, device_name="CPU")

    def infer(inputs):
        infer_request = compiled_model.create_infer_request()
        infer_request.infer(inputs, share_outputs=ov_model_params.share_outputs)
        outputs = [infer_request.get_output_tensor(i) for i in range(len(infer_request.results))]
        if ov_model_params.release_memory:
            compiled_model.release_memory()
        return outputs

    return infer


def _get_compress_decompress_model(
    config: WeightCompressionConfig,
    ov_model_params: OVModelParameters,
    parameters: List[ov._pyopenvino.op.Parameter],
    results: List[ov._pyopenvino.Node],
):
    if config.mode in [CompressWeightsMode.INT8_ASYM, config.mode.INT4_ASYM]:
        if len(results) == 1:
            compressed_w = results[0]
            s, zp = parameters[1], parameters[2]
        else:
            compressed_w, s, zp = results
        decompressed_w = (compressed_w - zp) * s
    else:
        if len(results) == 1:
            compressed_w = results[0]
            s = parameters[1]
        else:
            compressed_w, s = results
        decompressed_w = compressed_w * s

    model = ov.Model([decompressed_w], parameters)
    compiled_model = ov.compile_model(model, device_name="CPU")

    def infer(inputs):
        infer_request = compiled_model.create_infer_request()
        infer_request.infer(inputs, share_outputs=ov_model_params.share_outputs)
        outputs = [infer_request.get_output_tensor(i) for i in range(len(infer_request.results))]
        if ov_model_params.release_memory:
            compiled_model.release_memory()
        return outputs

    return infer
