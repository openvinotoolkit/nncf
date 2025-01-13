# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import UserDict
from typing import Any, Callable, Generator, Optional, Tuple, cast

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector


class StatisticPoint:
    """
    StatisticPoint stores information is necessary for statistics collection process:
    target_point from which statistics is collected: node_name and target_type determines the node edge.
    tensor_collector determines how to aggregate statistics in target_point
    algorithm implies on what algorithm nedeed this statistics.
    """

    def __init__(self, target_point: TargetPoint, tensor_collector: TensorCollector, algorithm: str):
        self.target_point = target_point
        self.algorithm_to_tensor_collectors = {algorithm: [tensor_collector]}

    def __eq__(self, other: Any) -> bool:
        return cast(
            bool,
            self.target_point == other.target_point
            and self.algorithm_to_tensor_collectors == other.self.algorithm_to_tensor_collectors,
        )


class StatisticPointsContainer(UserDict):  # type: ignore
    """
    Container with iteration interface for handling a composition of StatisticPoint.
    """

    def add_statistic_point(self, statistic_point: StatisticPoint) -> None:
        """
        Method to add statistic point to statistic point container.

        :param statistic_point: Statistic point to add.
        """
        target_node_name = statistic_point.target_point.target_node_name  # type: ignore
        if target_node_name not in self.data:
            self.data[target_node_name] = [statistic_point]
        else:
            for _statistic_point in self.data[target_node_name]:
                if _statistic_point.target_point == statistic_point.target_point:
                    for algorithm in statistic_point.algorithm_to_tensor_collectors:
                        if algorithm in _statistic_point.algorithm_to_tensor_collectors:
                            _statistic_point.algorithm_to_tensor_collectors[algorithm].extend(
                                statistic_point.algorithm_to_tensor_collectors[algorithm]
                            )
                        else:
                            _statistic_point.algorithm_to_tensor_collectors[algorithm] = (
                                statistic_point.algorithm_to_tensor_collectors[algorithm]
                            )
                    return

            self.data[target_node_name].append(statistic_point)

    def iter_through_statistic_points_in_target_node(
        self, target_node_name: str, filter_fn: Callable[[StatisticPoint], bool]
    ) -> Generator[StatisticPoint, None, None]:
        """
        Returns iterable through all statistic points in node with target_node_name.

        :param filter_fn: Callable to filter statistic containers according to its statistic point.
        :return: Iterable through all statistic points in node with target_node_name.
        """
        _statistic_points = self.data[target_node_name]
        for _statistic_point in _statistic_points:
            if filter_fn(_statistic_point):
                yield _statistic_point

    def get_tensor_collectors(
        self, filter_fn: Optional[Callable[[StatisticPoint], bool]] = None
    ) -> Generator[Tuple[str, StatisticPoint, TensorCollector], None, None]:
        """
        Returns iterable through all tensor collectors.

        :param filter_fn: Callable to filter statistic containers according to
            its statistic point. filter nothing by default.
        :return: Iterable through all tensor collectors in form of tuple of algorithm description,
            correspondent statistic point and tensor collector.
        """
        if filter_fn is None:

            def default_filter_fn(stat_point: StatisticPoint) -> bool:
                return True

            filter_fn = default_filter_fn

        for target_node_name in self.data:
            for statistic_point in self.iter_through_statistic_points_in_target_node(target_node_name, filter_fn):
                for algorithm, tensor_collectors in statistic_point.algorithm_to_tensor_collectors.items():
                    for tensor_collector in tensor_collectors:
                        yield algorithm, statistic_point, tensor_collector

    def get_algo_statistics_for_node(
        self,
        target_node_name: str,
        filter_fn: Callable[[StatisticPoint], bool],
        algorithm: str,
    ) -> Generator[TensorCollector, None, None]:
        """
        Returns iterable through all statistic collectors in node with target_node_name.

        :param filter_fn: Callable to filter statistic containers according to its statistic point.
        :return: Iterable through all statistic collectors in node with target_node_name.
        """
        for _statistic_point in self.iter_through_statistic_points_in_target_node(target_node_name, filter_fn):
            for _tensor_collector in _statistic_point.algorithm_to_tensor_collectors[algorithm]:
                yield _tensor_collector
