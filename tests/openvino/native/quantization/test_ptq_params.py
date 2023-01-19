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

from nncf.common.hardware.config import HW_CONFIG_TYPE_TARGET_DEVICE_MAP
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.experimental.openvino_native.graph.nncf_graph_builder import GraphConverter
from nncf.experimental.openvino_native.quantization.algorithms.min_max.openvino_backend import OVMinMaxAlgoBackend
from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector

from tests.openvino.native.models import LinearModel


# pylint: disable=protected-access
@pytest.mark.parametrize('target_device', [TargetDevice.CPU, TargetDevice.GPU, TargetDevice.VPU])
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = OVMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device.value == HW_CONFIG_TYPE_TARGET_DEVICE_MAP[target_device.value]


@pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
def test_range_type_per_tensor(range_type):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = OVMinMaxAlgoBackend()
    model = LinearModel().ov_model
    assert min_max_algo._parameters.range_type == range_type
    stat_points = min_max_algo.get_statistic_points(model)

    for _, stat_point in stat_points.items():
        for stat_point_ in stat_point:
            for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                if range_type is None:
                    # default tensor_collector for per-tensor
                    assert isinstance(tensor_collector, OVMeanMinMaxStatisticCollector)
                elif range_type == RangeType.MINMAX:
                    assert isinstance(tensor_collector, OVMinMaxStatisticCollector)
                elif range_type == RangeType.MEAN_MINMAX:
                    assert isinstance(tensor_collector, OVMeanMinMaxStatisticCollector)


@pytest.mark.parametrize('quantize_outputs', [False, True])
def test_quantize_outputs(quantize_outputs):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(quantize_outputs=quantize_outputs))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = OVMinMaxAlgoBackend()
    model = LinearModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    assert min_max_algo._parameters.quantize_outputs == quantize_outputs
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
    act_num_q, weight_num_q = 0, 0
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_activation_quantization_point():
            act_num_q += 1
        if quantization_point.is_weight_quantization_point():
            weight_num_q += 1

    if quantize_outputs:
        assert act_num_q == 3
    else:
        assert act_num_q == 1
    assert weight_num_q == 1


@pytest.mark.parametrize('ignored_scopes',
                         [[], ['MatMul'], ['Add'], ['MatMul', 'Add']],
                         ids=['empty', 'MatMul', 'Add', 'MatMul,Add'])
def test_ignored_scopes(ignored_scopes):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(ignored_scopes=ignored_scopes))
    min_max_algo = algo.algorithms[0]
    assert min_max_algo._parameters.ignored_scopes == ignored_scopes
    min_max_algo._backend_entity = OVMinMaxAlgoBackend()

    model = LinearModel().ov_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
    act_num_q, weight_num_q = 0, 0
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_activation_quantization_point():
            act_num_q += 1
        if quantization_point.is_weight_quantization_point():
            weight_num_q += 1

    if ignored_scopes == ['MatMul', 'Add']:
        assert act_num_q == 0
    else:
        assert act_num_q == 1
    assert weight_num_q == 1
