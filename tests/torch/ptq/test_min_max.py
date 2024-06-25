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
from typing import Tuple

import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.layer_attributes import ConstantLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.operator_metatypes import PTConstNoopMetatype
from nncf.torch.graph.operator_metatypes import PTConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTDepthwiseConv2dSubtype
from nncf.torch.graph.operator_metatypes import PTLinearMetatype
from nncf.torch.graph.transformations.commands import PTTargetPoint
from tests.post_training.test_templates.models import NNCFGraphToTest
from tests.post_training.test_templates.test_min_max import TemplateTestGetChannelAxes
from tests.post_training.test_templates.test_min_max import TemplateTestGetTargetPointShape
from tests.post_training.test_templates.test_min_max import TemplateTestMinMaxAlgorithm


class TestTorchMinMaxAlgorithm(TemplateTestMinMaxAlgorithm):
    @property
    def backend(self) -> MinMaxAlgoBackend:
        return PTMinMaxAlgoBackend

    @property
    def conv_metatype(self):
        return PTConv2dMetatype

    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> PTTargetPoint:
        if target_point_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        return PTTargetPoint(target_point_type, name, input_port_id=port_id)


class TestTorchGetTargetPointShape(TemplateTestGetTargetPointShape, TestTorchMinMaxAlgorithm):
    def get_nncf_graph(self, weight_port_id: int, weight_shape: Tuple[int]) -> NNCFGraph:
        conv_layer_attrs = ConvolutionLayerAttributes(
            weight_requires_grad=True,
            in_channels=weight_shape[1],
            out_channels=weight_shape[0],
            kernel_size=weight_shape[2:],
            stride=1,
            dilations=1,
            groups=1,
            transpose=False,
            padding_values=[],
        )
        return NNCFGraphToTest(
            PTConv2dMetatype,
            conv_layer_attrs,
            PTNNCFGraph,
            const_metatype=PTConstNoopMetatype,
            const_layer_attrs=ConstantLayerAttributes("w", shape=weight_shape),
        ).nncf_graph


class TestTorchGetChannelAxes(TemplateTestGetChannelAxes, TestTorchMinMaxAlgorithm):
    @property
    def depthwiseconv_metatype(self):
        return PTDepthwiseConv2dSubtype

    @property
    def matmul_metatype(self):
        return PTLinearMetatype

    @staticmethod
    def get_conv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> ConvolutionLayerAttributes:
        return ConvolutionLayerAttributes(
            weight_requires_grad=False,
            in_channels=weight_shape[0],
            out_channels=weight_shape[1],
            kernel_size=weight_shape[2:],
            stride=1,
            dilations=1,
            groups=1,
            transpose=False,
            padding_values=[],
        )

    @staticmethod
    def get_depthwiseconv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> ConvolutionLayerAttributes:
        return ConvolutionLayerAttributes(
            weight_requires_grad=False,
            in_channels=weight_shape[1],
            out_channels=weight_shape[2],
            kernel_size=weight_shape[3:],
            stride=1,
            dilations=1,
            groups=weight_shape[0],
            transpose=False,
            padding_values=[],
        )

    @staticmethod
    def get_matmul_node_attrs(weight_port_id: int, transpose_weight: Tuple[int], weight_shape: Tuple[int]):
        return LinearLayerAttributes(False, in_features=weight_shape[0], out_features=weight_shape[1])

    def test_get_channel_axes_matmul_node_ov_onnx(self):
        pytest.skip("Test is not applied for Torch backend.")

    def test_get_channel_axes_deptwiseconv_node_ov(self):
        pytest.skip("Test is not applied for Torch backend.")
