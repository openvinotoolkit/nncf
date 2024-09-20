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
import os
from typing import Optional, Tuple, List

import numpy as np
import openvino as ov
from openvino.runtime import opset13 as opset

import nncf
from nncf import CompressWeightsMode
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig


class OVCompressionPrimitiveCache:
    def __init__(self):
        self._compress_weight_model_cache = {}
        self._compress_weight_end_to_end_model_cache = {}
        self._compress_decompress_weight_model_cache = {}
        self._compress_decompress_end_to_end_weight_model_cache = {}

    def get_compress_weight_primitive_end_to_end(
        self,
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        reduction_axes: Optional[Tuple],
    ):
        DYNAMIC_COMPRESSION = bool(int(os.environ.get("DYNAMIC_COMPRESSION", "0")))
        if DYNAMIC_COMPRESSION:
            weight_shape = (-1,) * len(weight_shape)

        recompile = bool(int(os.environ.get("RECOMPILE", "0")))
        if recompile:
            return self._build_compress_model_end_to_end(config, weight_shape, reduction_axes)
        key = (config.mode, config.num_bits, weight_shape, reduction_axes)
        if key not in self._compress_weight_end_to_end_model_cache:
            self._compress_weight_end_to_end_model_cache[key] = self._build_compress_model_end_to_end(
                config, weight_shape, reduction_axes
            )
        return self._compress_weight_end_to_end_model_cache[key]

    def get_compress_weight_primitive(
        self,
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
    ):
        DYNAMIC_COMPRESSION = bool(int(os.environ.get("DYNAMIC_COMPRESSION", "0")))
        if DYNAMIC_COMPRESSION:
            weight_shape = (-1,) * len(weight_shape)
            scale_shape = (-1,) * (len(scale_shape) - 1) + (1,)
            if zero_point_shape is not None:
                zero_point_shape = (-1,) * (len(zero_point_shape) - 1) + (1,)

        recompile = bool(int(os.environ.get("RECOMPILE", "0")))
        if recompile:
            return self._build_compress_model(config, weight_shape, scale_shape, zero_point_shape)
        key = (config.mode, config.num_bits, weight_shape, scale_shape)
        if zero_point_shape is not None:
            key += (zero_point_shape,)
        if key not in self._compress_weight_model_cache:
            self._compress_weight_model_cache[key] = self._build_compress_model(
                config, weight_shape, scale_shape, zero_point_shape
            )
        return self._compress_weight_model_cache[key]

    def get_compress_decompress_weight_primitive(
        self,
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        reduction_axes: Optional[Tuple] = None,
        scale_shape: Optional[Tuple] = None,
        zero_point_shape: Optional[Tuple] = None,
    ):
        DYNAMIC_COMPRESSION = bool(int(os.environ.get("DYNAMIC_COMPRESSION", "0")))
        if DYNAMIC_COMPRESSION:
            weight_shape = (-1,) * len(weight_shape)
            if scale_shape is not None:
                scale_shape = (-1,) * (len(scale_shape) - 1) + (1,)
            if zero_point_shape is not None:
                zero_point_shape = (-1,) * (len(zero_point_shape) - 1) + (1,)

        recompile = bool(int(os.environ.get("RECOMPILE", "0")))
        if recompile:
            return self._build_compress_decompress_model(config, weight_shape, reduction_axes, scale_shape, zero_point_shape)
        key = (config.mode, config.num_bits, weight_shape)
        if reduction_axes is not None:
            key += (reduction_axes,)
        if scale_shape is not None:
            key += (scale_shape,)
        if zero_point_shape is not None:
            key += (zero_point_shape,)
        if key not in self._compress_decompress_weight_model_cache:
            self._compress_decompress_weight_model_cache[key] = self._build_compress_decompress_model(
                config, weight_shape, reduction_axes, scale_shape, zero_point_shape
            )
        return self._compress_decompress_weight_model_cache[key]

    @staticmethod
    def _build_compress_model_end_to_end(
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        reduction_axes: Optional[Tuple] = None,
        return_nodes: bool = False,
    ):
        INPUT_DTYPE = os.environ.get("INPUT_DTYPE", "fp32")

        if INPUT_DTYPE == "fp32":
            input_dtype = ov.Type.f32
        elif INPUT_DTYPE == "fp16":
            input_dtype = ov.Type.f16
        elif INPUT_DTYPE == "bf16":
            input_dtype = ov.Type.bf16
        else:
            raise Exception
        weight = opset.parameter(weight_shape, name="w", dtype=input_dtype)
        parameters = [weight]

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
                    f"Channel size {channel_size} should be divisible by size of group {group_size}")

            num_groups_per_channel = channel_size // group_size
            shape = list(weight.shape)  # [a1, r, a2] - "r" refers to number of channels along reduction axis
            shape[reduction_axes: reduction_axes + 1] = (num_groups_per_channel, group_size)
            weight = opset.reshape(weight, shape, special_zero=False)
            reduction_axes += 1

        mode = config.mode
        num_bits = config.num_bits
        eps = np.finfo(np.float32).eps
        if mode in [CompressWeightsMode.INT8_ASYM, CompressWeightsMode.INT4_ASYM]:
            min_values = opset.reduce_min(weight, reduction_axes=reduction_axes,
                                          keep_dims=True)  # [a1, r, a2] -> [a1, 1, a2]
            max_values = opset.reduce_max(weight, reduction_axes=reduction_axes,
                                          keep_dims=True)  # [a1, r, a2] -> [a1, 1, a2]
            min_values, max_values = opset.convert(min_values, ov.Type.f32), opset.convert(max_values, ov.Type.f32)

            level_low = 0
            level_high = 2 ** num_bits - 1
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

        return OVCompressionPrimitiveCache._get_compress_model(
            config,
            parameters,
            weight,
            scale,
            zero_point,
            output_only_weight=False,
            return_nodes=return_nodes,
        )

    @staticmethod
    def _build_compress_model(
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
        return_nodes: bool = False,
    ):
        INPUT_DTYPE = os.environ.get("INPUT_DTYPE", "fp32")

        if INPUT_DTYPE == "fp32":
            input_dtype = ov.Type.f32
        elif INPUT_DTYPE == "fp16":
            input_dtype = ov.Type.f16
        elif INPUT_DTYPE == "bf16":
            input_dtype = ov.Type.bf16
        else:
            raise Exception
        weight = opset.parameter(weight_shape, name="w", dtype=input_dtype)
        scale = opset.parameter(scale_shape, name="s", dtype=ov.Type.f32)
        parameters = [weight, scale]

        zero_point = None
        if config.mode in [CompressWeightsMode.INT8_ASYM, config.mode.INT4_ASYM]:
            zero_point = opset.parameter(zero_point_shape, name="zp", dtype=ov.Type.i32)
            parameters.append(zero_point)

        return OVCompressionPrimitiveCache._get_compress_model(
            config,
            parameters,
            weight,
            scale,
            zero_point,
            output_only_weight=True,
            return_nodes=return_nodes,
        )

    @staticmethod
    def _build_compress_decompress_model_end_to_end(
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        reduction_axes: Optional[Tuple] = None,
    ):
        parameters, results = OVCompressionPrimitiveCache._build_compress_model_end_to_end(
            config, weight_shape, reduction_axes, return_nodes=True
        )
        # `results` holds compressed weight, scale and, possibly, zero point
        return OVCompressionPrimitiveCache._get_compress_decompress_model(config, parameters, results)

    @staticmethod
    def _build_compress_decompress_model(
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
    ):
        parameters, results = OVCompressionPrimitiveCache._build_compress_model(
            config, weight_shape, scale_shape, zero_point_shape, return_nodes=True
        )
        # `results` holds only compressed weight
        return OVCompressionPrimitiveCache._get_compress_decompress_model(config, parameters, results)

    @staticmethod
    def _get_compress_model(
        config: WeightCompressionConfig,
        parameters: List[ov._pyopenvino.op.Parameter],
        w: ov.runtime.Node,
        s: ov.runtime.Node,
        zp: Optional[ov.runtime.Node] = None,
        output_only_weight: Optional[bool] = True,
        return_nodes: Optional[bool] = False,
    ):
        if w.get_element_type() != ov.Type.f32:
            w = opset.convert(w, ov.Type.f32)

        compressed_w = w / s

        num_bits = config.num_bits
        if config.mode in [CompressWeightsMode.INT8_ASYM, config.mode.INT4_ASYM]:
            dtype = ov.Type.u8
            level_low = 0
            level_high = 2**num_bits - 1
            compressed_w += opset.convert(zp, ov.Type.f32)
        elif config.mode in [CompressWeightsMode.INT8_SYM, config.mode.INT4_SYM]:
            dtype = ov.Type.i8
            level_low = -(2 ** (num_bits - 1))
            level_high = 2 ** (num_bits - 1) - 1
        else:
            raise Exception

        compressed_w = opset.clamp(opset.round(compressed_w), level_low, level_high, name="compressed_weights")

        FP32_OUTPUT = bool(int(os.environ.get("FP32_OUTPUT", "0")))
        if not FP32_OUTPUT:
            compressed_w = opset.convert(compressed_w, dtype)

        results = [compressed_w]
        if not output_only_weight:
            results.append(s)
            if zp is not None:
                results.append(opset.convert(zp, ov.Type.i32))
        if return_nodes:
            return parameters, results

        model = ov.Model(results, parameters)

        compiled_model = ov.compile_model(model, device_name="CPU")

        SHARE_OUTPUTS = bool(int(os.environ.get("SHARE_OUTPUTS", "0")))
        return compiled_model, lambda parameters: compiled_model(parameters, share_outputs=SHARE_OUTPUTS)

    @staticmethod
    def _get_compress_decompress_model(
        config: WeightCompressionConfig,
        parameters: List[ov._pyopenvino.op.Parameter],
        results: List[ov._pyopenvino.op.Parameter]
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

        return lambda parameters: compiled_model(parameters)[0]


OV_COMPRESSION_PRIMITIVE_CACHE = OVCompressionPrimitiveCache()
