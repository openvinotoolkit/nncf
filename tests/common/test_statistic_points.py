# Copyright (c) 2024 Intel Corporation
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

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer


class TestStatisticPointsContainer:
    @pytest.fixture
    def container_with_statistics(self):
        container = StatisticPointsContainer()
        target_type = TargetType.LAYER
        target_point = TargetPoint(target_type)
        target_point.target_node_name = "Node"
        # tensor_collector = TensorStatisticCollectorBase()
        statistic_point = StatisticPoint(target_point, TensorStatisticCollectorBase, "minmax")
        container.add_statistic_point(statistic_point)
        return container

    def test_remove_statistic_points(self, container_with_statistics):
        container_with_statistics.remove_statistic_points("minimax")

        # Verify that the statistics for the specified algorithm are removed
        for target_node_name, statistic_points in container_with_statistics.data.items():
            for statistic_point in statistic_points:
                assert "minimax" not in statistic_point.algorithm_to_tensor_collectors

    def test_remove_statistic_points_empty_container(self):
        empty_container = StatisticPointsContainer()
        empty_container.remove_statistic_points("algorithm")

        # Verify that removing statistics from an empty container has no effect
        assert not empty_container.data
