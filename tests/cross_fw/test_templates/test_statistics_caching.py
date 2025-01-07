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
from abc import abstractmethod
from pathlib import Path

import pytest

import nncf
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.tensor import Tensor


class TemplateTestStatisticsCaching:
    @property
    @abstractmethod
    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> TargetPoint:
        """
        Creates a backend-specific TargetPoint.

        :param target_point_type: The type of target point (e.g., PRE_LAYER_OPERATION).
        :param name: The name of the target point.
        :param port_id: The port ID for the target point.
        :return: A backend-specific TargetPoint.
        """
        pass

    @abstractmethod
    def get_statistics_aggregator(self):
        """
        Returns a statistics aggregator. Must be implemented by subclasses.

        :return: Statistics aggregator instance specific to the backend.
        """
        pass

    @abstractmethod
    def _create_dummy_min_max_tensor(self) -> Tensor:
        """
        Creates a dummy tensor for testing purposes.

        :return: A Tensor object with dummy data.
        """

    def _create_dummy_statistic_point(self) -> StatisticPoint:
        """
        Creates a dummy statistic point for testing purposes.

        :return: A StatisticPoint object with dummy data.
        """
        dummy_t_p = self.create_target_point(TargetType.PRE_LAYER_OPERATION, "dummy_name", 0)
        dummy_tensor_collector = TensorCollector()
        dummy_tensor_collector._cached_statistics = MinMaxTensorStatistic(*self._create_dummy_min_max_tensor())
        return StatisticPoint(
            target_point=dummy_t_p, tensor_collector=dummy_tensor_collector, algorithm="dummy_algorithm"
        )

    def test_dump_and_load_statistics(self, tmp_path: Path):
        """
        Tests the dumping and loading of statistics to and from a file.

        :param tmp_path: The temporary path provided by pytest.
        """
        test_dir = "test_dir"
        aggregator = self.get_statistics_aggregator()
        statistics_points = StatisticPointsContainer()

        dummy_statistic_point = self._create_dummy_statistic_point()
        statistics_points.add_statistic_point(dummy_statistic_point)

        aggregator.statistic_points = statistics_points
        aggregator.dump_statistics(tmp_path / test_dir)
        assert (tmp_path / test_dir).exists(), "Statistics file was not created"

        aggregator.load_statistics_from_dir(tmp_path / test_dir)

    def test_incorrect_backend_statistics_load(self, tmp_path: Path):
        """
        Tests the dumping and loading of statistics to and from a file with non matched backends.

        :param tmp_path: The temporary path provided by pytest.
        """
        test_file = "test"
        aggregator = self.get_statistics_aggregator()
        statistics_points = StatisticPointsContainer()

        dummy_statistic_point = self._create_dummy_statistic_point()
        statistics_points.add_statistic_point(dummy_statistic_point)

        aggregator.statistic_points = statistics_points
        aggregator.dump_statistics(tmp_path / test_file)
        assert (tmp_path / test_file).exists(), "Statistics file was not created"
        # spoil backend
        aggregator.BACKEND = BackendType.TENSORFLOW
        with pytest.raises(nncf.StatisticsCacheError):
            aggregator.load_statistics_from_dir(tmp_path / test_file)
