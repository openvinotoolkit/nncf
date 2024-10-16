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
from abc import abstractmethod
from pathlib import Path

import numpy as np

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.tensor import Tensor


class TemplateTestStatisticsCaching:
    @property
    @abstractmethod
    def create_target_point(self, target_point_type: TargetType, name: str, port_id: int) -> TargetPoint:
        "Creates backend specific TargetPoint."

    @abstractmethod
    def get_statistics_aggregator(self, dataset):
        """_summary_

        :param _type_ dataset: _description_
        """

    def _create_dummy_statistic_point(self):
        dummy_t_p = self.create_target_point(TargetType.PRE_LAYER_OPERATION, "dummy_name", 0)
        dummy_tensor_collector = TensorCollector()
        dummy_tensor_collector.statistics = MinMaxTensorStatistic(Tensor(np.zeros((3))), Tensor(np.ones((3))))
        return StatisticPoint(
            target_point=dummy_t_p, tensor_collector=dummy_tensor_collector, algorithm="dummy_algorithm"
        )

    def test_dump_and_load_statistics(self, tmp_path):
        test_file = "test"
        aggregator = self.get_statistics_aggregator()
        statistics_points = StatisticPointsContainer()

        dummy_statistic_point = self._create_dummy_statistic_point()
        statistics_points.add_statistic_point(dummy_statistic_point)

        aggregator.statistic_points = statistics_points
        aggregator.dump_statistics(tmp_path / test_file)
        assert Path(tmp_path / test_file).exists()
        aggregator.load_statistics_from_file(tmp_path / test_file)

    def test_dump_statistics(self, tmp_path):
        test_file = "test"
        aggregator = self.get_statistics_aggregator()
        statistics_points = StatisticPointsContainer()

        dummy_statistic_point = self._create_dummy_statistic_point()
        statistics_points.add_statistic_point(dummy_statistic_point)

        aggregator.statistic_points = statistics_points
        aggregator.dump_statistics(tmp_path / test_file)
        assert Path(tmp_path / test_file).exists()
