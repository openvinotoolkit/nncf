
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
from torch import nn

from nncf.scopes import IgnoredScope
from nncf.parameters import TargetDevice
from nncf.common.graph.patterns import GraphPattern
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from tests.post_training.test_ptq_params import TemplateTestPTQParams
from nncf.torch.tensor_statistics.collectors import PTMinMaxStatisticCollector
from nncf.torch.tensor_statistics.collectors import PTMeanMinMaxStatisticCollector

from tests.torch.helpers import create_bn, create_conv, create_depthwise_conv
from tests.torch.ptq.helpers import get_single_conv_nncf_graph
from tests.torch.ptq.helpers import get_single_no_weigth_matmul_nncf_graph
from tests.torch.ptq.helpers import get_nncf_network

# pylint: disable=protected-access


class ToNNCFNetworkInterface:
    def get_nncf_network(self):
        return get_nncf_network(self)


class LinearTestModel(nn.Module, ToNNCFNetworkInterface):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(3, 3, 1)
        self.bn1 = create_bn(3)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = create_conv(3, 1, 1)
        self.bn2 = create_bn(1)

    def forward(self, x):
        # input_shape = [1, 3, 32, 32]
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.avg_pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        return x


class OneDepthwiseConvModel(nn.Module, ToNNCFNetworkInterface):
    def __init__(self) -> None:
        super().__init__()
        self.depthwise_conv = create_depthwise_conv(3, 1, 1, 1)

    def forward(self, x):
        # input_shape = [1, 3, 32, 32]
        return self.depthwise_conv(x)


@pytest.mark.parametrize('target_device', TargetDevice)
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = PTMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device == target_device


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return PTMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return PTMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return PTMeanMinMaxStatisticCollector

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
            {'model': LinearTestModel().get_nncf_network(),
             'stat_points_num': 5},
        'test_range_type_per_channel':
            {'model': OneDepthwiseConvModel().get_nncf_network(),
             'stat_points_num': 2},
        'test_quantize_outputs':
            {'nncf_graph': get_single_conv_nncf_graph().nncf_graph,
             'pattern': GraphPattern()},
        'test_ignored_scopes':
            {'nncf_graph': get_single_conv_nncf_graph().nncf_graph,
             'pattern': GraphPattern()},
        'test_model_type_pass':
            {'nncf_graph': get_single_no_weigth_matmul_nncf_graph().nncf_graph,
             'pattern': GraphPattern()},
        }

    @pytest.fixture(params=[(IgnoredScope([]), 1, 1),
                            (IgnoredScope(['/Conv_1_0']), 0, 0)])
    def ignored_scopes_data(self, request):
        return request.param
