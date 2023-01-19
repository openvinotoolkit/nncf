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
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from nncf import Dataset
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.quantization.algorithms.definitions import RangeType
from nncf.experimental.openvino_native.statistics.aggregator import OVStatisticsAggregator
from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint

from tests.openvino.native.models import LinearModel

INPUT_SHAPE = [1, 3, 4, 2]

DATASET_SAMPLES = [np.zeros(INPUT_SHAPE), np.ones(INPUT_SHAPE)]

DATASET_SAMPLES[0][0, 0, 0, 0] = 1  # max
DATASET_SAMPLES[0][0, 0, 0, 1] = -10  # min

DATASET_SAMPLES[0][0, 1, 0, 0] = 0.1  # max
DATASET_SAMPLES[0][0, 1, 0, 1] = -1  # min

DATASET_SAMPLES[0][0, 2, 0, 0] = 128  # max
DATASET_SAMPLES[0][0, 2, 0, 1] = -128  # min


@dataclass
class TestParameters:
    range_type: RangeType
    use_abs_max: bool
    reduction_shape: Optional[Tuple[int]]
    ref_max_val: Union[np.ndarray, float]
    ref_min_val: Union[np.ndarray, float]


@pytest.mark.parametrize('test_parameters',
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
    target_node_name = 'Input'
    model = LinearModel().ov_model
    dataset = Dataset(DATASET_SAMPLES, transform_func=lambda data: {target_node_name: data})

    statistics_aggregator = OVStatisticsAggregator(dataset)
    statistics_points = StatisticPointsContainer()
    if test_parameters.range_type == RangeType.MINMAX:
        tensor_collector = OVMinMaxStatisticCollector(test_parameters.use_abs_max, test_parameters.reduction_shape,
                                                        num_samples=len(DATASET_SAMPLES))
    if test_parameters.range_type == RangeType.MEAN_MINMAX:
        tensor_collector = OVMeanMinMaxStatisticCollector(False, test_parameters.use_abs_max,
                                                            test_parameters.reduction_shape,
                                                            num_samples=len(DATASET_SAMPLES))
    algorithm_name = 'TestAlgo'
    statistic_point_type = TargetType.POST_LAYER_OPERATION
    target_point = OVTargetPoint(statistic_point_type, target_node_name, 0)
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
