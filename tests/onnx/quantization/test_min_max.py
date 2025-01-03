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
from typing import Tuple

import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetType
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXGemmMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.test_min_max import MATMUL_WEIGHT_SHAPE
from tests.cross_fw.test_templates.test_min_max import TemplateTestGetChannelAxes
from tests.cross_fw.test_templates.test_min_max import TemplateTestGetTargetPointShape
from tests.cross_fw.test_templates.test_min_max import TemplateTestMinMaxAlgorithm


class TestONNXMinMaxAlgorithm(TemplateTestMinMaxAlgorithm):
    @property
    def backend(self) -> MinMaxAlgoBackend:
        return ONNXMinMaxAlgoBackend

    @property
    def conv_metatype(self):
        return ONNXConvolutionMetatype

    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_point_type, name, port_id)


class TestONNXGetTargetPointShape(TemplateTestGetTargetPointShape, TestONNXMinMaxAlgorithm):
    def get_nncf_graph(self, weight_port_id: int, weight_shape: Tuple[int]) -> NNCFGraph:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={weight_port_id: {"shape": weight_shape}}, bias_attrs={})
        return NNCFGraphToTest(ONNXConvolutionMetatype, conv_layer_attrs).nncf_graph


class TestONNXGetChannelAxesMinMaxAlgorithm(TemplateTestGetChannelAxes, TestONNXMinMaxAlgorithm):
    @property
    def depthwiseconv_metatype(self):
        return ONNXDepthwiseConvolutionMetatype

    @property
    def matmul_metatype(self):
        return ONNXGemmMetatype

    @staticmethod
    def get_conv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> ONNXLayerAttributes:
        return ONNXLayerAttributes(weight_attrs={weight_port_id: {"shape": weight_shape}}, bias_attrs={})

    @staticmethod
    def get_depthwiseconv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> ONNXLayerAttributes:
        return TestONNXGetChannelAxesMinMaxAlgorithm.get_conv_node_attrs(weight_port_id, weight_shape)

    @staticmethod
    def get_matmul_node_attrs(
        weight_port_id: int, transpose_weight: Tuple[int], weight_shape: Tuple[int]
    ) -> ONNXLayerAttributes:
        weight_attrs = {weight_port_id: {"name": "dummy", "shape": weight_shape}}
        if weight_port_id == 0:
            gemm_attrs = {"transA": int(transpose_weight), "transB": 0}
        elif weight_port_id == 1:
            gemm_attrs = {"transA": 0, "transB": int(transpose_weight)}
        return ONNXLayerAttributes(weight_attrs=weight_attrs, node_attrs=gemm_attrs)

    def test_get_channel_axes_deptwiseconv_node_ov(self):
        pytest.skip("Test is not applied for ONNX backend.")

    def test_get_channel_axes_matmul_torch(self):
        pytest.skip("Test is not applied for ONNX backend.")

    @pytest.mark.parametrize(
        "weight_shape, weight_port_id, transpose_weight, ref_axes",
        (
            (MATMUL_WEIGHT_SHAPE, 1, False, (-1,)),
            (MATMUL_WEIGHT_SHAPE, 1, True, (-2,)),
            (MATMUL_WEIGHT_SHAPE, 0, True, (-1,)),
            (MATMUL_WEIGHT_SHAPE, 0, False, (-2,)),
        ),
    )
    def test_get_channel_axes_matmul_node_ov_onnx(self, weight_shape, weight_port_id, transpose_weight, ref_axes):
        super().test_get_channel_axes_matmul_node_ov_onnx(weight_shape, weight_port_id, transpose_weight, ref_axes)
