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
from abc import ABC
from abc import abstractmethod
from itertools import islice
from typing import Dict, TypeVar, Any

from tqdm import tqdm

from nncf.common.factory import ModelTransformerFactory
from nncf.common.factory import EngineFactory
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.data.dataset import Dataset

TensorType = TypeVar('TensorType')
TModel = TypeVar('TModel')


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.stat_subset_size = 0
        self.statistic_points = StatisticPointsContainer()

    def collect_statistics(self, model: TModel) -> None:
        """
        Collects statistics for registered StatisticPoints.
        The statistics are stored in self.statistic_points.

        :param model: backend-specific model instance
        """
        model_transformer = ModelTransformerFactory.create(model)

        transformation_layout = self._get_transformation_layout_extra_outputs(self.statistic_points)
        model_with_outputs = model_transformer.transform(transformation_layout)
        engine = EngineFactory.create(model_with_outputs)

        for input_data in tqdm(islice(self.dataset.get_inference_data(), self.stat_subset_size),
                               total=self.stat_subset_size):
            outputs = engine.infer(input_data)
            processed_outputs = self._process_outputs(outputs)
            self._register_statistics(processed_outputs, self.statistic_points)

    def register_stastistic_points(self, statistic_points: StatisticPointsContainer) -> None:
        """
        Register statistic points for statistics collection and recalculates the maximum number samples
        for collecting statistics, based on the maximum value from the all algorithms.

        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """
        for _, _statistic_points in statistic_points.items():
            for _statistic_point in _statistic_points:
                self.statistic_points.add_statistic_point(_statistic_point)

        for _, _statistic_points in self.statistic_points.items():
            for _statistic_point in _statistic_points:
                for _, tensor_collectors in _statistic_point.algorithm_to_tensor_collectors.items():
                    for tensor_collector in tensor_collectors:
                        self.stat_subset_size = max(
                            self.stat_subset_size, tensor_collector.num_samples)

    @abstractmethod
    def _register_statistics(self,
                             outputs: Dict[str, NNCFTensor],
                             statistic_points: StatisticPointsContainer) -> None:
        """
        Process prepared raw model outputs and statistic points for the further usage.

        :param outputs: prepared raw model outputs
        :param statistic_points: StatisticPointsContainer instance with the statistic points
        """

    @staticmethod
    @abstractmethod
    def _get_transformation_layout_extra_outputs(statistic_points: StatisticPointsContainer) -> TransformationLayout:
        """
        Create backend-specific transformation layout for the further statistics collection

        :param statistic_points: StatisticPointsContainer to add outputs
        :return: TransformationLayout with the corresponding transformations
        """

    @staticmethod
    @abstractmethod
    def _process_outputs(outputs: Any) -> Dict[str, NNCFTensor]:
        """
        Post-process model outputs for the further statistics collection.

        :param outputs: raw model outputs
        :return: processed model outputs in Dict[str, NNCFTensor] format
        """
