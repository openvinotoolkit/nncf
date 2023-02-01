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

from typing import Callable, Generator

from collections import UserDict

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor import TensorType
from nncf.common.graph.transformations.commands import TargetPoint


class StatisticPoint:
    """
    StatisticPoint stores information is necessary for statistics collection process:
    target_point from which statistics is collected: node_name and target_type determines the node edge.
    tensor_collector determines how to aggregate statistics in target_point
    algorithm implies on what algorithm nedeed this statistics.
    """

    def __init__(self, target_point: TargetPoint, tensor_collector: TensorStatisticCollectorBase,
                 algorithm: 'Algorithm'):
        self.target_point = target_point
        self.algorithm_to_tensor_collectors = {algorithm: [tensor_collector]}

    def __eq__(self, other):
        if self.target_point == other.target_point and \
                self.algorithm_to_tensor_collectors == other.self.algorithm_to_tensor_collectors:
            return True
        return False

    def register_tensor(self, x: TensorType):
        for tensor_collectors in self.algorithm_to_tensor_collectors.values():
            for tensor_collector in tensor_collectors:
                tensor_collector.register_input(x)


class StatisticPointsContainer(UserDict):
    """
    Container with iteration interface for handling a composition of StatisticPoint.
    """

    def add_statistic_point(self, statistic_point: StatisticPoint):
        target_node_name = statistic_point.target_point.target_node_name
        if target_node_name not in self.data:
            self.data[target_node_name] = [statistic_point]
        else:
            for _statistic_point in self.data[target_node_name]:
                if _statistic_point.target_point == statistic_point.target_point:
                    for algorithm in statistic_point.algorithm_to_tensor_collectors.keys():
                        if algorithm in _statistic_point.algorithm_to_tensor_collectors:
                            _statistic_point.algorithm_to_tensor_collectors[algorithm].extend(
                                statistic_point.algorithm_to_tensor_collectors[algorithm])
                            return
                        _statistic_point.algorithm_to_tensor_collectors[
                            algorithm] = statistic_point.algorithm_to_tensor_collectors[algorithm]
                        return
            self.data[target_node_name].append(statistic_point)

    def iter_through_statistic_points_in_target_node(self, target_node_name: str,
                                                     statistic_point_condition_func: Callable[
                                                         [StatisticPoint], bool]):
        _statistic_points = self.data[target_node_name]
        for _statistic_point in _statistic_points:
            if statistic_point_condition_func(_statistic_point):
                yield _statistic_point

    def get_algo_statistics_for_node(
            self,
            target_node_name: str,
            statistic_point_condition_func: Callable[[StatisticPoint], bool],
            algorithm: 'Algorithm') -> Generator[TensorStatisticCollectorBase, None, None]:

        for _statistic_point in self.iter_through_statistic_points_in_target_node(target_node_name,
                                                                                  statistic_point_condition_func):
            for _tensor_collector in _statistic_point.algorithm_to_tensor_collectors[algorithm]:
                yield _tensor_collector
