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

from typing import Dict, Iterator

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.tensor import TensorType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase


class StatisticPoint:
    """
    StatisticPoint stores information is necessary for statistics collection process:
    target_point from which statistics is collected: node_name and target_type determines the node edge.
    tensor_collector determines how to aggregate statistics in target_point
    algorithm implies on what algorithm nedeed this statistics.
    """

    def __init__(
        self, target_point: TargetPoint, tensor_collector: TensorStatisticCollectorBase, tensor_collector_key: str
    ):
        """
        :param target_point: Specifies where statistic should be collected.
        :param tensor_collector: Tensor collector.
        :param tensor_collector_key: Key of tensor collector. Should be unique
            for each tensor collector inside statistic point.
        """
        self._target_point = target_point
        self._tensor_collectors = {tensor_collector_key: tensor_collector}

    @property
    def target_point(self) -> TargetPoint:
        return self._target_point

    @property
    def tensor_collectors(self) -> Dict[str, TensorStatisticCollectorBase]:
        return self._tensor_collectors

    def get_tensor_collector(self, tensor_collector_key: str) -> TensorStatisticCollectorBase:
        """
        Returns tensor collector associated with provided key.

        :param tensor_collector_key: Key of tensor collector.
        :return: A tensor collector associated with provided key.
        """
        return self._tensor_collectors[tensor_collector_key]

    def register_tensor(self, x: TensorType):
        for tensor_collector in self.tensor_collectors.values():
            tensor_collector.register_input(x)

    def unite(self, statistic_point: "StatisticPoint") -> None:
        """
        Unites `self` and `statistic_point` statistic points into `self`.

        :param statistic_point: Statistic point to unite with `self`.
        """
        if self.target_point != statistic_point.target_point:
            raise ValueError("Unable to unite statistic points with different target points.")

        intersection_of_keys = self._tensor_collectors.keys() & statistic_point.tensor_collectors.keys()
        if len(intersection_of_keys) != 0:
            raise ValueError(f"Unable to unite statistic points with common keys: {list(intersection_of_keys)}")

        self._tensor_collectors.update(statistic_point.tensor_collectors)


class StatisticPointsContainer:
    """
    Container with iteration interface for handling a composition of StatisticPoint.
    """

    def __init__(self):
        self._target_point_to_statistic_point = {}

    def add_statistic_point(self, statistic_point: StatisticPoint) -> None:
        """
        Adds statistic point to container.

        :param statistic_point: Statistic point to add.
        """
        target_point = statistic_point.target_point

        if target_point in self._target_point_to_statistic_point:
            self._target_point_to_statistic_point[target_point].unite(statistic_point)
        else:
            self._target_point_to_statistic_point[target_point] = statistic_point

    def get_statistic_point(self, target_point: TargetPoint) -> StatisticPoint:
        """
        Returns statistic point associated with provided target point.

        :param target_point: Target point
        :return: Statistic point associated with provided target point.
        """
        return self._target_point_to_statistic_point[target_point]

    def __iter__(self) -> Iterator[StatisticPoint]:
        return iter(self._target_point_to_statistic_point.values())
