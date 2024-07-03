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

from typing import Optional, Tuple

import openvino as ov
from openvino.runtime import opset13 as opset

from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig


class OVCompressionPrimitiveCache:
    def __init__(self):
        self._compress_weight_model_cache = {}
        self._compress_decompress_weight_model_cache = {}

    def get_compress_weight_primitive(
        self,
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
        invert_scale: Optional[bool] = False,
    ):
        key = (config.mode, config.num_bits, weight_shape, scale_shape, invert_scale)
        if zero_point_shape is not None:
            key += (zero_point_shape,)
        if key not in self._compress_weight_model_cache:
            self._compress_weight_model_cache[key] = self._build_compress_model(
                config, weight_shape, scale_shape, zero_point_shape, invert_scale
            )
        return self._compress_weight_model_cache[key]

    def get_compress_decompress_weight_primitive(
        self,
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
    ):
        key = (config.mode, config.num_bits, weight_shape, scale_shape)
        if zero_point_shape is not None:
            key += (zero_point_shape,)
        if key not in self._compress_decompress_weight_model_cache:
            self._compress_decompress_weight_model_cache[key] = self._build_compress_decompress_model(
                config, weight_shape, scale_shape, zero_point_shape
            )
        return self._compress_decompress_weight_model_cache[key]

    @staticmethod
    def _build_compress_model(
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
        invert_scale: Optional[bool] = False,
        return_nodes: bool = False,
    ):
        w = opset.parameter(weight_shape, name="w")
        s = opset.parameter(scale_shape, name="s")
        parameters = [w, s]
        if invert_scale:
            compressed_w = w * (1 / s)
        else:
            compressed_w = w / s
        num_bits = config.num_bits
        if zero_point_shape is not None:
            level_low = 0
            level_high = 2**num_bits - 1

            zp = opset.parameter(zero_point_shape, name="zp")
            parameters.append(zp)
            compressed_w += zp
        else:
            level_low = -(2 ** (num_bits - 1))
            level_high = 2 ** (num_bits - 1) - 1

        result = opset.clamp(opset.round(compressed_w), level_low, level_high, name="compressed_weights")

        if return_nodes:
            return parameters, result

        model = ov.Model([result], parameters)

        compiled_model = ov.compile_model(model, device_name="CPU")

        return lambda parameters: compiled_model(parameters)[0]

    @staticmethod
    def _build_compress_decompress_model(
        config: WeightCompressionConfig,
        weight_shape: Tuple,
        scale_shape: Tuple,
        zero_point_shape: Optional[Tuple] = None,
    ):
        parameters, clamp = OVCompressionPrimitiveCache._build_compress_model(
            config, weight_shape, scale_shape, zero_point_shape, return_nodes=True
        )

        if len(parameters) == 3:
            _, s, zp = parameters
            result = (clamp - zp) * s
        else:
            s = parameters[1]
            result = clamp * s

        model = ov.Model([result], parameters)
        compiled_model = ov.compile_model(model, device_name="CPU")

        return lambda parameters: compiled_model(parameters)[0]


OV_COMPRESSION_PRIMITIVE_CACHE = OVCompressionPrimitiveCache()
