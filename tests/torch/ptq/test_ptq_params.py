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
from torch import nn

from nncf.common.graph.patterns import GraphPattern
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.backend import BackendType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.scopes import IgnoredScope
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.operator_metatypes import PTModuleConv2dMetatype
from nncf.torch.graph.operator_metatypes import PTModuleLinearMetatype
from nncf.torch.graph.operator_metatypes import PTSoftmaxMetatype
from nncf.torch.tensor_statistics.collectors import PTMeanMinMaxStatisticCollector
from nncf.torch.tensor_statistics.collectors import PTMinMaxStatisticCollector
from tests.common.quantization.metatypes import Conv2dTestMetatype
from tests.common.quantization.metatypes import LinearTestMetatype
from tests.common.quantization.metatypes import SoftmaxTestMetatype
from tests.post_training.test_ptq_params import TemplateTestPTQParams
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv
from tests.torch.helpers import create_depthwise_conv
from tests.torch.ptq.helpers import get_nncf_network
from tests.torch.ptq.helpers import get_single_conv_nncf_graph
from tests.torch.ptq.helpers import get_single_no_weight_matmul_nncf_graph

# pylint: disable=protected-access


def get_hw_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_hw_pattern_graph(backend=BackendType.TORCH, device=device)


def get_ignored_patterns(device: TargetDevice = TargetDevice.ANY) -> GraphPattern:
    return PatternsManager.get_full_ignored_pattern_graph(backend=BackendType.TORCH, device=device)


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


@pytest.mark.parametrize("target_device", TargetDevice)
def test_target_device(target_device):
    algo = PostTrainingQuantization(target_device=target_device)
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = PTMinMaxAlgoBackend()
    assert min_max_algo._target_device == target_device


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return PTMinMaxAlgoBackend()

    def check_is_min_max_statistic_collector(self, tensor_collector):
        assert isinstance(tensor_collector, PTMinMaxStatisticCollector)

    def check_is_mean_min_max_statistic_collector(self, tensor_collector):
        assert isinstance(tensor_collector, PTMeanMinMaxStatisticCollector)

    def check_quantize_outputs_fq_num(self, quantize_outputs, act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 2
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    def target_point(self, target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @property
    def metatypes_mapping(self):
        return {
            Conv2dTestMetatype: PTModuleConv2dMetatype,
            LinearTestMetatype: PTModuleLinearMetatype,
            SoftmaxTestMetatype: PTSoftmaxMetatype,
        }

    @pytest.fixture(scope="session")
    def test_params(self):
        return {
            "test_range_estimator_per_tensor": {"model": LinearTestModel().get_nncf_network(), "stat_points_num": 5},
            "test_range_estimator_per_channel": {
                "model": OneDepthwiseConvModel().get_nncf_network(),
                "stat_points_num": 2,
            },
            "test_quantize_outputs": {
                "nncf_graph": get_single_conv_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_ignored_scopes": {
                "nncf_graph": get_single_conv_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
            "test_model_type_pass": {
                "nncf_graph": get_single_no_weight_matmul_nncf_graph().nncf_graph,
                "hw_patterns": get_hw_patterns(),
                "ignored_patterns": get_ignored_patterns(),
            },
        }

    @pytest.fixture(params=[(IgnoredScope([]), 1, 1), (IgnoredScope(["/Conv_1_0"]), 0, 0)])
    def ignored_scopes_data(self, request):
        return request.param
