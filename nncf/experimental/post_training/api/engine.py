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

from abc import ABC, abstractmethod
from typing import Callable, Dict, TypeVar

from tqdm import tqdm

from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.post_training.api.dataset import NNCFData
from nncf.experimental.post_training.api.sampler import Sampler

ModelType = TypeVar('ModelType')
MetricType = TypeVar('MetricType')


class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """
    # TODO (Nikita Malinin): Update class with the _get_sampler() method
    def __init__(self):
        self.model = None
        self._sampler = None
        self._dataset = None
        self._metrics = None
        self._inputs_transforms = lambda input_data: input_data
        self._outputs_transforms = lambda output_data: output_data

    # TODO (Nikita Malinin): Add statistic aggregator object (per-backend)
    @property
    def dataset(self):
        return self._dataset

    @property
    def metrics(self):
        return self._metrics

    @property
    def sampler(self):
        return self._sampler

    def set_dataset(self, dataset) -> None:
        self._dataset = dataset

    def set_metrics(self, metrics) -> None:
        self._metrics = metrics

    def set_sampler(self, sampler) -> None:
        self._sampler = sampler

    def set_model(self, model: ModelType) -> None:
        self.model = model

    def is_model_set(self) -> bool:
        return self.model is not None

    def get_sampler(self) -> Sampler:
        if not self.sampler:
            raise RuntimeError(f'The {self.__class__} tried to get sampler, but it was not set')
        return self.sampler

    def set_outputs_transforms(self, outputs_transforms: Callable):
        """
        Sets outputs transforms that applies to the model outputs after inference.
        """
        self._outputs_transforms = outputs_transforms

    def set_inputs_transforms(self, inputs_transforms: Callable):
        """
        Sets inputs transforms that applies to the input before inference.
        """
        self._inputs_transforms = inputs_transforms

    def compute_statistics(self, statistic_points: StatisticPointsContainer) -> None:
        """
        Performs model inference on specified dataset subset and collects statistics

        :param statistic_points: StatisticPointsContaine with StatisticPoints,
         in which statistics are collected and registered.
        """
        if not self.is_model_set():
            raise RuntimeError(f'The {self.__class__} tried to compute statistics, '
                               'while the model was not set.')
        sampler = self.get_sampler()

        for input_data in tqdm(sampler):
            outputs = self.infer(input_data)
            self._register_statistics(outputs, statistic_points)

    def compute_metrics(self, metrics_per_sample: bool = False) -> Dict[str, MetricType]:
        """
        Performs model inference on specified dataset subset for metrics calculation

        :param metrics_per_sample: whether to collect metrics for each batch
        :return metrics: a tuple of dictionaries of persample and
        overall metric values if 'metrics_per_sample' is True
        """
        if not self.is_model_set():
            raise RuntimeError(f'The {self.__class__} tried to compute statistics, '
                               'while the model was not set.')

        # TODO (Nikita Malinin): Add per-sample metrics calculation
        sampler = self.get_sampler()
        for input_data in sampler:
            outputs = self.infer(input_data)
            self.metrics.update(outputs, input_data)
        return self.metrics.avg_value

    @abstractmethod
    def infer(self, input_data: NNCFData) -> NNCFData:
        """
        Runs model on the provided input_data.
        Returns the dictionary of model outputs by node names.

        :param input_data: inputs for the model transformed with the inputs_transforms
        :return output_data: models output after outputs_transforms
        """

    @abstractmethod
    def _register_statistics(self, outputs: NNCFData, statistic_points: StatisticPointsContainer) -> None:
        """
        Does mapping from the provided output and statistics_points to register statistics.
        """
