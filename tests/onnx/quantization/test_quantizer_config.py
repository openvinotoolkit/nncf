"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest

from nncf.common.graph.transformations.commands import TargetType
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.onnx.graph.nncf_graph_builder import ONNXExtendedLayerAttributes
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXAddLayerMetatype

from tests.post_training.test_quantizer_config import TemplateTestQuantizerConfig
from tests.post_training.models import NNCFGraphToTest
from tests.post_training.models import NNCFGraphToTestDepthwiseConv
from tests.post_training.models import NNCFGraphToTestSumAggregation


ParamsCls = TemplateTestQuantizerConfig.TestGetStatisticsCollectorParameters
class TestQuantizerConfig(TemplateTestQuantizerConfig):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return ONNXMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return ONNXMeanMinMaxStatisticCollector

    @pytest.fixture(params=[pytest.param((TargetType.PRE_LAYER_OPERATION, '/Sum_1_0', (0, 2), (0, 1, 2)),
                                         marks=pytest.mark.skip(
                                             'Ticket 102414: remove hardcoded axes for activations')),
                            (TargetType.POST_LAYER_OPERATION, '/Conv_1_0', (0, 2, 3), None),
                            (TargetType.OPERATION_WITH_WEIGHTS,  '/Conv_1_0', (1, 2, 3), None)])
    def statistic_collector_parameters(self, request) -> ParamsCls:
        return ParamsCls(*request.param)

    @pytest.fixture
    def single_conv_nncf_graph(self) -> NNCFGraphToTest:
        conv_layer_attrs = ONNXExtendedLayerAttributes('dummy', 'dummy', (4, 4, 4, 4))
        return NNCFGraphToTest(ONNXConvolutionMetatype, conv_layer_attrs)

    @pytest.fixture
    def depthwise_conv_nncf_graph(self) -> NNCFGraphToTestDepthwiseConv:
        return NNCFGraphToTestDepthwiseConv(ONNXDepthwiseConvolutionMetatype,
                                            ONNXExtendedLayerAttributes('dummy', 'dummy'))

    @pytest.fixture
    def conv_sum_aggregation_nncf_graph(self) ->\
        NNCFGraphToTestSumAggregation:
        conv_layer_attrs = ONNXExtendedLayerAttributes('dummy', 'dummy', (4, 4, 4, 4))
        return NNCFGraphToTestSumAggregation(ONNXConvolutionMetatype, ONNXAddLayerMetatype,
                                             conv_layer_attrs)
