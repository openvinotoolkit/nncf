# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nncf.common.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.operator_metatypes import OutputNoopMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from tests.common.quantization.mock_graphs import NodeWithType
from tests.common.quantization.test_filter_constant_nodes import create_mock_graph
from tests.common.quantization.test_filter_constant_nodes import get_nncf_graph_from_mock_nx_graph
from tests.post_training.test_templates.test_quantizer_config import TemplateTestQuantizerConfig

ParamsCls = TemplateTestQuantizerConfig.TestGetStatisticsCollectorParameters


class NNCFGraphToTest:
    def __init__(self, conv_metatype, conv_layer_attrs=None, nncf_graph_cls=NNCFGraph):
        #       Original graph
        #          Input_1
        #             |
        #           Conv_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=ONNXLayerAttributes()),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=ONNXLayerAttributes()),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)


class NNCFGraphToTestDepthwiseConv:
    def __init__(self, depthwise_conv_metatype, conv_layer_attrs=None):
        #       Original graph
        #          Input_1
        #             |
        #        DepthwiseConv_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=ONNXLayerAttributes()),
            NodeWithType("Conv_1", depthwise_conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=ONNXLayerAttributes()),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph)


class NNCFGraphToTestSumAggregation:
    def __init__(self, conv_metatype, sum_metatype, conv_layer_attrs=None, nncf_graph_cls=NNCFGraph):
        #       Original graph
        #          Input_1
        #             |
        #          Conv_1
        #             |
        #           Sum_1
        #             |
        #           Output_1
        nodes = [
            NodeWithType("Input_1", InputNoopMetatype, layer_attributes=ONNXLayerAttributes()),
            NodeWithType("Conv_1", conv_metatype, layer_attributes=conv_layer_attrs),
            NodeWithType("Sum_1", sum_metatype, layer_attributes=ONNXLayerAttributes()),
            NodeWithType("Output_1", OutputNoopMetatype, layer_attributes=ONNXLayerAttributes()),
        ]
        node_edges = [("Input_1", "Conv_1"), ("Conv_1", "Sum_1"), ("Sum_1", "Output_1")]
        original_mock_graph = create_mock_graph(nodes, node_edges)
        self.nncf_graph = get_nncf_graph_from_mock_nx_graph(original_mock_graph, nncf_graph_cls)
        # Hack output size of the Sum_1 operation
        self.nncf_graph._nx_graph.out_edges[("2 /Sum_1_0", "3 /Output_1_0")][
            self.nncf_graph.ACTIVATION_SHAPE_EDGE_ATTR
        ] = [1, 1, 1]


class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    def check_is_min_max_statistic_collector(self, tensor_collector):
        assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)

    def check_is_mean_min_max_statistic_collector(self, tensor_collector):
        assert isinstance(tensor_collector, ONNXMeanMinMaxStatisticCollector)

    @pytest.fixture(
        params=[
            pytest.param(
                (TargetType.PRE_LAYER_OPERATION, "/Sum_1_0", (0, 2), (0, 1, 2)),
                marks=pytest.mark.skip("Ticket 102414: remove hardcoded axes for activations"),
            ),
            (TargetType.POST_LAYER_OPERATION, "/Conv_1_0", (0, 2, 3), None),
            (TargetType.OPERATION_WITH_WEIGHTS, "/Conv_1_0", (1, 2, 3), None),
        ]
    )
    def statistic_collector_parameters(self, request) -> ParamsCls:
        return ParamsCls(*request.param)

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={})
        return NNCFGraphToTest(ONNXConvolutionMetatype, conv_layer_attrs)

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return NNCFGraphToTestDepthwiseConv(
            ONNXDepthwiseConvolutionMetatype,
            ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={}),
        )

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) -> NNCFGraphToTestSumAggregation:
        conv_layer_attrs = ONNXLayerAttributes(weight_attrs={1: {"shape": [4, 4, 4, 4]}}, bias_attrs={})
        return NNCFGraphToTestSumAggregation(ONNXConvolutionMetatype, ONNXAddLayerMetatype, conv_layer_attrs)
