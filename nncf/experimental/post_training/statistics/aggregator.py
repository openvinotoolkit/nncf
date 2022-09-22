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
from nncf.experimental.post_training.api.sampler import Sampler
from nncf.experimental.post_training.graph.model_transformer import StaticModelTransformerBase

TensorType = TypeVar('TensorType')
ModelType = TypeVar('ModelType')


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, engine: Engine, dataset: Dataset, model_transformer: StaticModelTransformerBase):
        self.engine = engine
        self.dataset = dataset
        self.is_calculate_metric = False
        self.max_number_samples = 0
        self.statistic_points = StatisticPointsContainer()
        self.model_transformer = model_transformer

    def collect_statistics(self, model: ModelType) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: input model for the statistics collection
        """
        self.model_transformer.set_model(model)
        self.engine.set_model(self.model_transformer.prepare_model_for_statistics_collection(self.statistic_points))
        self.engine.set_sampler(self._create_sampler(self.dataset, self.max_number_samples))
        self.engine.compute_statistics(self.statistic_points)

    def register_stastistic_points(self, statistic_points: StatisticPointsContainer):
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.
        """
        for _, _statistic_points in statistic_points.items():
            for _statistic_point in _statistic_points:
                self.statistic_points.add_statistic_point(_statistic_point)

        for _, _statistic_points in self.statistic_points.items():
            for _statistic_point in _statistic_points:
                for _, tensor_collectors in _statistic_point.algorithm_to_tensor_collectors.items():
                    for tensor_collector in tensor_collectors:
                        self.max_number_samples = max(self.max_number_samples, tensor_collector.num_samples)

    @abstractmethod
    def _create_sampler(self, dataset: Dataset,
                        sample_indices: int) -> Sampler:
        """
        Create backend-specific Sampler.
        """
