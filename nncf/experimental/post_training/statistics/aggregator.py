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

from abc import ABC
from abc import abstractmethod

from typing import TypeVar

from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataset import Dataset

TensorType = TypeVar('TensorType')
ModelType = TypeVar('ModelType')


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, engine: Engine, dataset: Dataset):
        self.engine = engine
        self.dataset = dataset
        self.is_calculate_metric = False
        self.max_number_samples = 10
        self.statistic_points = StatisticPointsContainer()

    @abstractmethod
    def collect_statistics(self, model: ModelType) -> None:
        """
        Collects statistics for layers determined in self.layers_statistics.
        The statistics are stored in self.layers_statistics.
        """

    def register_stastistic_points(self, stastistic_points: StatisticPointsContainer):
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.
        """
        for target_point, stastistic_point in stastistic_points.items():
            self.statistic_points.add_statistic_point(stastistic_point)

        for _, statistic_point in self.statistic_points.items():
            for algorithm, tensor_collector in statistic_point.algorithm_to_tensor_collector.items():
                if tensor_collector.num_samples is not None:
                    self.max_number_samples = max(self.max_number_samples, tensor_collector.num_samples)
