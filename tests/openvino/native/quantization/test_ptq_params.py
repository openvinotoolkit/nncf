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
from nncf.common.utils.backend import BackendType
from nncf.common.graph.patterns.manager import PatternsManager
from nncf.common.graph.patterns import GraphPattern
from nncf.parameters import TargetDevice
from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.openvino_native.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector

from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import DepthwiseConv4DModel
from tests.post_training.test_ptq_params import TemplateTestPTQParams


def get_patterns_setup() -> GraphPattern:
    backend = BackendType.OPENVINO
    device = TargetDevice.ANY
    return PatternsManager.get_full_pattern_graph(backend, device)


# pylint: disable=protected-access
@pytest.mark.parametrize('target_device', [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.VPU])
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = OVMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device.value == HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device.value]


class TestPTQParams(TemplateTestPTQParams):
    def get_algo_backend(self):
        return OVMinMaxAlgoBackend()

    def get_min_max_statistic_collector_cls(self):
        return OVMinMaxStatisticCollector

    def get_mean_max_statistic_collector_cls(self):
        return OVMeanMinMaxStatisticCollector

    def check_quantize_outputs_fq_num(self, quantize_outputs,
                                      act_num_q, weight_num_q):
        if quantize_outputs:
            assert act_num_q == 3
        else:
            assert act_num_q == 1
        assert weight_num_q == 1

    @pytest.fixture(scope='session')
    def test_params(self):
        return {
        'test_range_type_per_tensor':
            {'model': LinearModel().ov_model,
             'stat_points_num': 2},
        'test_range_type_per_channel':
            {'model': DepthwiseConv4DModel().ov_model,
             'stat_points_num': 2},
        'test_quantize_outputs':
            {'nncf_graph': GraphConverter.create_nncf_graph(LinearModel().ov_model),
             'pattern': get_patterns_setup()},
        'test_ignored_scopes':
            {'nncf_graph': GraphConverter.create_nncf_graph(LinearModel().ov_model),
             'pattern': get_patterns_setup()},
        }

    @pytest.fixture(params=[(IgnoredScope(), 1, 1), (IgnoredScope(['MatMul']), 1, 0),
                            (IgnoredScope(['Add']), 1, 1),
                            (IgnoredScope(['MatMul', 'Add']), 0, 0)])
    def ignored_scopes_data(self, request):
        return request.param
