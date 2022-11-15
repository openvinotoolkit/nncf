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
from typing import Callable, TypeVar, Dict


from nncf.common.tensor import NNCFTensor
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer

TModel = TypeVar('TModel')
ModelInput = TypeVar('ModelInput')

class Engine(ABC):
    """
    The basic class aims to provide the interface to infer the model.
    """

    def __init__(self):
        self.model = None
        self._dataset = None
        self._inputs_transforms = lambda input_data: input_data
        self._outputs_transforms = lambda output_data: output_data

    # TODO (Nikita Malinin): Add statistic aggregator object (per-backend)
    @property
    def dataset(self):
        return self._dataset

    def set_dataset(self, dataset) -> None:
        self._dataset = dataset

    def set_model(self, model: TModel) -> None:
        self.model = model

    def is_model_set(self) -> bool:
        return self.model is not None

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

    @abstractmethod
    def compute_statistics(self, statistic_points: StatisticPointsContainer, subset_size: int = None) -> None:
        """
        Performs model inference on specified dataset subset and collects statistics

        :param statistic_points: StatisticPointsContaine with StatisticPoints,
         in which statistics are collected and registered.
        """

    @abstractmethod
    def infer(self, input_data: ModelInput) -> Dict[str, NNCFTensor]:
        """
        Runs model on the provided input_data.
        Returns the dictionary of model outputs by node names.

        :param input_data: inputs for the model transformed with the inputs_transforms
        :return output_data: models output after outputs_transforms
        """

    @abstractmethod
    def _register_statistics(self, outputs: Dict[str, NNCFTensor], statistic_points: StatisticPointsContainer) -> None:
        """
        Does mapping from the provided output and statistics_points to register statistics.
        """
