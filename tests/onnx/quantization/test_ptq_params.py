"""
 Copyright (c) 2022 Intel Corporation
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
from nncf.parameters import TargetDevice
from nncf.experimental.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.quantization.algorithms.definitions import RangeType
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.min_max.onnx_backend import \
    ONNXMinMaxAlgoBackend
from tests.onnx.models import LinearModel
from tests.onnx.models import OneDepthwiseConvolutionalModel
from tests.onnx.quantization.test_quantizer_config import NNCFGraphToTest


# pylint: disable=protected-access

@pytest.mark.parametrize('target_device', TargetDevice)
def test_target_device(target_device):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(target_device=target_device))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    assert min_max_algo._parameters.target_device == target_device


@pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
@pytest.mark.parametrize('original_model', [LinearModel()])
def test_range_type_per_tensor(range_type, original_model):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    model = original_model.onnx_model
    assert min_max_algo._parameters.range_type == range_type
    stat_points = min_max_algo.get_statistic_points(model)

    for _, stat_point in stat_points.items():
        for stat_point_ in stat_point:
            for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                if range_type is None:
                    # default tensor_collector for per-tensor
                    assert isinstance(tensor_collector, ONNXMeanMinMaxStatisticCollector)
                if range_type == RangeType.MINMAX:
                    assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)
                elif range_type == RangeType.MEAN_MINMAX:
                    assert isinstance(tensor_collector, ONNXMeanMinMaxStatisticCollector)


@pytest.mark.parametrize('range_type', [RangeType.MINMAX, RangeType.MEAN_MINMAX, None])
@pytest.mark.parametrize('original_model', [OneDepthwiseConvolutionalModel()])
def test_range_type_per_channel(range_type, original_model):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(range_type=range_type))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    model = original_model.onnx_model
    assert min_max_algo._parameters.range_type == range_type
    stat_points = min_max_algo.get_statistic_points(model)

    for _, stat_point in stat_points.items():
        for stat_point_ in stat_point:
            for tensor_collector in stat_point_.algorithm_to_tensor_collectors[MinMaxQuantization]:
                # Range_type does not affect per-channel tensor_collector
                if range_type is None:
                    assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)
                if range_type == RangeType.MINMAX:
                    assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)
                elif range_type == RangeType.MEAN_MINMAX:
                    assert isinstance(tensor_collector, ONNXMinMaxStatisticCollector)


@pytest.mark.parametrize('quantize_outputs', [False, True])
def test_quantize_outputs(quantize_outputs):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(quantize_outputs=quantize_outputs))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    nncf_graph = NNCFGraphToTest().nncf_graph
    assert min_max_algo._parameters.quantize_outputs == quantize_outputs
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
    act_num_q, weight_num_q = 0, 0
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_activation_quantization_point():
            act_num_q += 1
        if quantization_point.is_weight_quantization_point():
            weight_num_q += 1

    if quantize_outputs:
        assert act_num_q == 2
    else:
        assert act_num_q == 1
    assert weight_num_q == 1


@pytest.mark.parametrize('ignored_scopes', [[], ['/Conv_1_0']])
def test_ignored_scopes(ignored_scopes):
    algo = PostTrainingQuantization(PostTrainingQuantizationParameters(ignored_scopes=ignored_scopes))
    min_max_algo = algo.algorithms[0]
    min_max_algo._backend_entity = ONNXMinMaxAlgoBackend()
    nncf_graph = NNCFGraphToTest().nncf_graph
    assert min_max_algo._parameters.ignored_scopes == ignored_scopes
    q_setup = min_max_algo._get_quantizer_setup(nncf_graph)
    act_num_q, weight_num_q = 0, 0
    for quantization_point in q_setup.quantization_points.values():
        if quantization_point.is_activation_quantization_point():
            act_num_q += 1
        if quantization_point.is_weight_quantization_point():
            weight_num_q += 1

    if ignored_scopes:
        assert act_num_q == 0
    else:
        assert act_num_q == 1
    assert weight_num_q == 1
