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
from abc import abstractmethod
from typing import Tuple

import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import BaseLayerAttributes
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend

CONV_WEIGHT_SHAPE = (3, 10, 4, 4)
DEPTHWISECONV_WEIGHT_SHAPE = (5, 10, 20, 7, 7)
MATMUL_WEIGHT_SHAPE = (2, 4)


class TemplateTestMinMaxAlgorithm:
    @property
    @abstractmethod
    def backend(self) -> MinMaxAlgoBackend:
        """
        Get backend specific MinMaxAlgoBackend

        :return MinMaxAlgoBackend: Backend specific MinMaxAlgoBackend
        """

    @property
    @abstractmethod
    def conv_metatype(self):
        "Backend specific Convolution metatype."

    @property
    @abstractmethod
    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> TargetPoint:
        "Creates backend specific TargetPoint."


class TemplateTestGetTargetPointShape(TemplateTestMinMaxAlgorithm):
    @abstractmethod
    def get_nncf_graph(self, weight_port_id: int, weight_shape: Tuple[int]) -> NNCFGraph:
        "Returns backend specific NNCFGraph having a single Convolution."

    @pytest.mark.parametrize(
        "target_point_type, input_port_id, reference_shape",
        (
            (TargetType.PRE_LAYER_OPERATION, 0, (1, 3, 224, 224)),
            (TargetType.POST_LAYER_OPERATION, 0, (1, 10, 224, 224)),
            (TargetType.OPERATION_WITH_WEIGHTS, 1, (3, 10, 4, 4)),
        ),
    )
    def test_get_target_point_shape(
        self, target_point_type: TargetType, input_port_id: int, reference_shape: Tuple[int]
    ):
        nncf_graph = self.get_nncf_graph(input_port_id, CONV_WEIGHT_SHAPE)
        nodes = nncf_graph.get_nodes_by_metatypes((self.conv_metatype,))
        assert len(nodes) == 1
        node = nodes.pop()
        target_point = self.create_target_point(target_point_type, node.node_name, input_port_id)
        assert self.backend().get_target_point_shape(nncf_graph, node, target_point) == reference_shape


class TemplateTestGetChannelAxes(TemplateTestMinMaxAlgorithm):
    @property
    @abstractmethod
    def depthwiseconv_metatype(self):
        "Backend specific Depthwise convolution metatype."

    @property
    @abstractmethod
    def matmul_metatype(self):
        "Backend specific MatMul metatype."

    @staticmethod
    @abstractmethod
    def get_conv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> BaseLayerAttributes:
        "Returns backend specific layer attributes for Convolution."

    @staticmethod
    @abstractmethod
    def get_depthwiseconv_node_attrs(weight_port_id: int, weight_shape: Tuple[int]) -> BaseLayerAttributes:
        "Returns backend specific layer attributes for Convolution."

    @staticmethod
    @abstractmethod
    def get_matmul_node_attrs(
        weight_port_id: int, transpose_weight: Tuple[int], weight_shape: Tuple[int]
    ) -> BaseLayerAttributes:
        "Returns backend specific layer attributes for MatMul."

    @pytest.mark.parametrize(
        "conv_shape, weight_port_id, ref_axes", ((CONV_WEIGHT_SHAPE, 0, (0,)), (CONV_WEIGHT_SHAPE, 1, (0,)))
    )
    def test_get_channel_axes_conv_node(self, conv_shape, weight_port_id, ref_axes):
        """
        Checks Convolution quantization axes in MinMax for OV, ONNX and Torch.
        """
        conv_node = NNCFNode({"metatype": self.conv_metatype})
        conv_node.layer_attributes = self.get_conv_node_attrs(weight_port_id, conv_shape)
        target_point = self.create_target_point(TargetType.PRE_LAYER_OPERATION, None, weight_port_id)
        assert self.backend().get_weight_quantization_axes(conv_node, target_point, len(conv_shape)) == ref_axes

    @pytest.mark.parametrize(
        "conv_shape, weight_port_id, ref_axes",
        ((DEPTHWISECONV_WEIGHT_SHAPE, 0, (0,)), (DEPTHWISECONV_WEIGHT_SHAPE, 1, (0,))),
    )
    def test_get_channel_axes_deptwiseconv_node_onnx_torch(self, conv_shape, weight_port_id, ref_axes):
        """
        Checks Depthwise convolution quantization axes in MinMax for ONNX and Torch.
        """
        conv_node = NNCFNode({"metatype": self.depthwiseconv_metatype})
        conv_node.layer_attributes = self.get_depthwiseconv_node_attrs(weight_port_id, conv_shape)
        target_point = self.create_target_point(TargetType.PRE_LAYER_OPERATION, None, weight_port_id)
        assert self.backend().get_weight_quantization_axes(conv_node, target_point, len(conv_shape)) == ref_axes

    @pytest.mark.parametrize(
        "conv_shape, weight_port_id, ref_axes",
        ((DEPTHWISECONV_WEIGHT_SHAPE, 0, (0, 1)), (DEPTHWISECONV_WEIGHT_SHAPE, 1, (0, 1))),
    )
    def test_get_channel_axes_deptwiseconv_node_ov(self, conv_shape, weight_port_id, ref_axes):
        """
        Checks Depthwise convolution quantization axes in MinMax for OV.
        """
        conv_node = NNCFNode({"metatype": self.depthwiseconv_metatype})
        conv_node.layer_attributes = self.get_depthwiseconv_node_attrs(weight_port_id, conv_shape)
        target_point = self.create_target_point(TargetType.PRE_LAYER_OPERATION, None, weight_port_id)
        assert self.backend().get_weight_quantization_axes(conv_node, target_point, len(conv_shape)) == ref_axes

    @pytest.mark.parametrize(
        "weight_shape, weight_port_id, transpose_weight, ref_axes",
        (
            (MATMUL_WEIGHT_SHAPE, 1, False, (1,)),
            (MATMUL_WEIGHT_SHAPE, 1, True, (0,)),
            (MATMUL_WEIGHT_SHAPE, 0, True, (1,)),
            (MATMUL_WEIGHT_SHAPE, 0, False, (0,)),
        ),
    )
    def test_get_channel_axes_matmul_node_ov_onnx(self, weight_shape, weight_port_id, transpose_weight, ref_axes):
        """
        Checks MatMul quantization axes in MinMax for OV and ONNX.
        """
        matmul_node = NNCFNode({"metatype": self.matmul_metatype})
        matmul_node.layer_attributes = self.get_matmul_node_attrs(weight_port_id, transpose_weight, weight_shape)
        target_point = self.create_target_point(TargetType.PRE_LAYER_OPERATION, None, weight_port_id)
        assert self.backend().get_weight_quantization_axes(matmul_node, target_point, len(weight_shape)) == ref_axes

    @pytest.mark.parametrize(
        "weight_shape, ref_axes",
        # Torch has strict specification - weight has the following layout: [C_OUT, C_IN]
        ((MATMUL_WEIGHT_SHAPE, (0,)),),
    )
    def test_get_channel_axes_matmul_torch(self, weight_shape, ref_axes):
        """
        Checks MatMul quantization axes in MinMax for Torch.
        """
        matmul_node = NNCFNode({"metatype": self.matmul_metatype})

        class DummyTargetPoint:
            input_port_id = 0

        assert (
            self.backend().get_weight_quantization_axes(matmul_node, DummyTargetPoint(), len(weight_shape)) == ref_axes
        )
