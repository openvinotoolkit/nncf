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
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from tests.cross_fw.test_templates.models import NNCFGraphToTest
from tests.cross_fw.test_templates.test_min_max import TemplateTestGetChannelAxes
from tests.cross_fw.test_templates.test_min_max import TemplateTestGetTargetPointShape
from tests.cross_fw.test_templates.test_min_max import TemplateTestMinMaxAlgorithm


class TestOVMinMaxAlgorithm(TemplateTestMinMaxAlgorithm):
    @property
    def backend(self) -> MinMaxAlgoBackend:
        return OVMinMaxAlgoBackend

    @property
    def conv_metatype(self):
        return OVConvolutionMetatype

    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_point_type, name, port_id)


class TestOVGetTargetPointShape(TemplateTestGetTargetPointShape, TestOVMinMaxAlgorithm):
    def get_nncf_graph(self, weight_port_id: int, weight_shape: Tuple[int]) -> NNCFGraph:
        conv_layer_attrs = OVLayerAttributes({weight_port_id: {"name": "dummy", "shape": weight_shape, "dtype": "f32"}})
        return NNCFGraphToTest(OVConvolutionMetatype, conv_layer_attrs).nncf_graph


class TestOVGetChannelAxes(TemplateTestGetChannelAxes, TestOVMinMaxAlgorithm):
    @property
    def depthwiseconv_metatype(self):
        return OVDepthwiseConvolutionMetatype

    @property
    def matmul_metatype(self):
        return OVMatMulMetatype

    @staticmethod
    def get_conv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> OVLayerAttributes:
        constant_attributes = {weight_port_id: {"name": "dummy", "shape": weight_shape, "dtype": "f32"}}
        return OVLayerAttributes(constant_attributes, {}, {})

    @staticmethod
    def get_depthwiseconv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> OVLayerAttributes:
        return TestOVGetChannelAxes.get_conv_node_attrs(weight_port_id, weight_shape)

    @staticmethod
    def get_matmul_node_attrs(
        weight_port_id: int, transpose_weight: Tuple[int], weight_shape: Tuple[int]
    ) -> OVLayerAttributes:
        constant_attributes = {weight_port_id: {"name": "dummy", "shape": weight_shape, "dtype": "f32"}}
        constant_attributes[weight_port_id]["transpose"] = transpose_weight
        return OVLayerAttributes(constant_attributes, {}, {})

    def test_get_channel_axes_deptwiseconv_node_onnx_torch(self):
        pytest.skip("Test is not applied for OV backend.")

    def test_get_channel_axes_matmul_torch(self):
        pytest.skip("Test is not applied for OV backend.")
