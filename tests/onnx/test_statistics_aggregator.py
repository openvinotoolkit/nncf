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

import numpy as np

from nncf.onnx.statistics.aggregator import ONNXStatisticsAggregator
from nncf.quantization.algorithms.definitions import RangeType
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.common.graph.transformations.commands import TargetType

from tests.onnx.models import InputOutputModel
from tests.onnx.quantization.common import get_dataset_for_test

INPUT_SHAPE = [3, 3, 3]

DATASET_SAMPLES = [np.zeros(INPUT_SHAPE), np.ones(INPUT_SHAPE)]

DATASET_SAMPLES[0][0, 0, 0] = 1  # max
DATASET_SAMPLES[0][0, 0, 1] = -10  # min

DATASET_SAMPLES[0][1, 0, 0] = 0.1  # max
DATASET_SAMPLES[0][1, 0, 1] = -1  # min

DATASET_SAMPLES[0][2, 0, 0] = 128  # max
DATASET_SAMPLES[0][2, 0, 1] = -128  # min


class TestParameters:
    def __init__(self, range_type, use_abs_max, reduction_shape, ref_max_val, ref_min_val):
        self.range_type = range_type
        self.use_abs_max = use_abs_max
        self.reduction_shape = reduction_shape
        self.ref_max_val = ref_max_val
        self.ref_min_val = ref_min_val


@pytest.mark.parametrize('test_parameters, ',
                         ((TestParameters(RangeType.MEAN_MINMAX, False, None, 64.5, -63.5)),
                          (TestParameters(RangeType.MEAN_MINMAX, False, (0, 2, 3), np.array((1, 0.55, 64.5)),
                                          np.array((-4.5, 0, -63.5)))),
                          (TestParameters(RangeType.MEAN_MINMAX, True, (0, 2, 3), np.array((5.5, 1, 64.5)),
                                          np.array((-4.5, 0, -63.5)))),
                          (TestParameters(RangeType.MINMAX, False, None, 128, -128)),
                          (TestParameters(RangeType.MINMAX, True, None, 128, -128)),
                          (TestParameters(RangeType.MINMAX, False, (0, 2, 3), np.array((1, 1, 128)),
                                          np.array((-10, -1, -128)))),
                          (TestParameters(RangeType.MINMAX, True, (0, 2, 3), np.array((10, 1, 128)),
                                          np.array((-10, -1, -128)))),
                          )
                         )
def test_statistics_aggregator(test_parameters):
    model = InputOutputModel().onnx_model

    dataset = get_dataset_for_test(DATASET_SAMPLES, "X")

    statistics_aggregator = ONNXStatisticsAggregator(dataset)
    statistics_points = StatisticPointsContainer()
    if test_parameters.range_type == RangeType.MINMAX:
        tensor_collector = ONNXMinMaxStatisticCollector(test_parameters.use_abs_max, test_parameters.reduction_shape,
                                                        num_samples=len(DATASET_SAMPLES))
    if test_parameters.range_type == RangeType.MEAN_MINMAX:
        tensor_collector = ONNXMeanMinMaxStatisticCollector(False, test_parameters.use_abs_max,
                                                            test_parameters.reduction_shape,
                                                            num_samples=len(DATASET_SAMPLES))
    target_node_name = 'Identity'
    algorithm_name = 'TestAlgo'
    statistic_point_type = TargetType.POST_LAYER_OPERATION
    target_point = ONNXTargetPoint(statistic_point_type, target_node_name, 0)
    statistics_points.add_statistic_point(StatisticPoint(target_point=target_point,
                                                         tensor_collector=tensor_collector,
                                                         algorithm=algorithm_name))
    statistics_aggregator.register_stastistic_points(statistics_points)
    statistics_aggregator.collect_statistics(model)

    def filter_func(point):
        return algorithm_name in point.algorithm_to_tensor_collectors and \
               point.target_point.type == statistic_point_type

    for tensor_collector in statistics_points.get_algo_statistics_for_node(
            target_node_name,
            filter_func,
            algorithm_name):
        stat = tensor_collector.get_statistics()
        assert np.allclose(stat.max_values, test_parameters.ref_max_val)
        assert np.allclose(stat.min_values, test_parameters.ref_min_val)
