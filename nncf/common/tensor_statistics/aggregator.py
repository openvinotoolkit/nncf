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

from nncf import Dataset
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.engine import Engine
from nncf.common.graph.model_transformer import ModelTransformer

TensorType = TypeVar('TensorType')
TModel = TypeVar('TModel')


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, engine: Engine, dataset: Dataset):
        self.engine = engine
        self.dataset = dataset
        self.is_calculate_metric = False
        self.max_number_samples = 0
        self.statistic_points = StatisticPointsContainer()

    def collect_statistics(self, model_transformer: ModelTransformer) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model_transformer: ModelTransformer intance with the model
        """
        transformation_layout = self._get_transformation_layout_extra_outputs(self.statistic_points)
        model_with_outputs = model_transformer.transform(transformation_layout)
        self.engine.set_model(model_with_outputs)
        self.engine.set_dataset(self.dataset)
        self.engine.compute_statistics(self.statistic_points, self.max_number_samples)

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
    def _get_transformation_layout_extra_outputs(
            self,
            statistic_points: StatisticPointsContainer) -> TransformationLayout:
        """
        Create backend-specific transformation layout for the further statistics collection

        :param statistic_points: StatisticPointsContainer to add outputs
        :return: TransformationLayout with the corresponding transformations
        """
