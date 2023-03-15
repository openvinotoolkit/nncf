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

from nncf.scopes import IgnoredScope
from nncf.parameters import TargetDevice
from nncf.common.graph.patterns import GraphPattern
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.onnx_backend import ONNXMinMaxAlgoBackend
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXLinearMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXExtendedLayerAttributes

from tests.onnx.models import LinearModel
from tests.onnx.models import OneDepthwiseConvolutionalModel
from tests.post_training.test_ptq_params import TemplateTestPTQParams
from tests.post_training.models import NNCFGraphToTest
from tests.post_training.models import NNCFGraphToTestMatMul


# pylint: disable=protected-access

@pytest.mark.parametrize('target_device', TargetDevice)
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device == target_device


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return ONNXMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return ONNXMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return ONNXMeanMinMaxStatisticCollector

    def check_quantize_outputs_fq_num(self, quantize_outputs,
                                      act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 2
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    @pytest.fixture(scope='session')
    def test_params(self):
        return {
            'test_range_type_per_tensor':
                {'model': LinearModel().onnx_model,
                 'stat_points_num': 5},
            'test_range_type_per_channel':
                {'model': OneDepthwiseConvolutionalModel().onnx_model,
                 'stat_points_num': 2},
            'test_quantize_outputs':
                {'nncf_graph': NNCFGraphToTest(ONNXConvolutionMetatype,
                                               ONNXExtendedLayerAttributes(None, None)).nncf_graph,
                 'pattern': GraphPattern()},
            'test_ignored_scopes':
                {'nncf_graph': NNCFGraphToTest(ONNXConvolutionMetatype,
                                               ONNXExtendedLayerAttributes(None, None)).nncf_graph,
                 'pattern': GraphPattern()},
            'test_model_type_pass':
                {'nncf_graph': NNCFGraphToTestMatMul(ONNXLinearMetatype).nncf_graph,
                 'pattern': GraphPattern()},
        }

    @pytest.fixture(params=[(IgnoredScope([]), 1, 1),
                            (IgnoredScope(['/Conv_1_0']), 0, 0)])
    def ignored_scopes_data(self, request):
        return request.param
