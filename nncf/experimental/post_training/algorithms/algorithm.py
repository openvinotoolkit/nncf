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
from typing import Dict
from typing import Union

from enum import Enum
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.post_training.api.engine import Engine
from nncf.common.utils.backend import BackendType

ModelType = TypeVar('ModelType')


class PostTrainingAlgorithms(Enum):
    MinMaxQuantization = 'min_max_quantization'
    FastBiasCorrection = 'fast_bias_correction'
    DefaultQuantization = 'default_quantization'


class AlgorithmParameters(ABC):
    """
    Base class for Post-Training algorithm parameters.
    """

    @abstractmethod
    def to_json(self) -> Dict[str, Union[str, float, int]]:
        """
        Serializes algorithm parameters to JSON format.
        """


class Algorithm(ABC):
    """
    Base class for all Post-Training algorithms.
    """

    @property
    @abstractmethod
    def available_backends(self) -> Dict[str, BackendType]:
        """
        Returns dictionary of the avaliable backends for the algorithm

        :return: Dict of backends supported by the algorithm
        """

    def apply(self, model: ModelType, engine: Engine,
              statistic_points: StatisticPointsContainer) -> ModelType:
        """
        Checks that statistic point exists, sets model into transformer
        and applies the algorithm to the model.
        :param model: model for applying algorithm
        :param engine: engine for the model execution
        :param statistic_points: StatisticPointsContainer
        :return: model after algorithm
        """
        _statistic_points = self.get_statistic_points(model)
        for edge_name in _statistic_points.keys():
            if statistic_points.get(edge_name) is None:
                raise RuntimeError(f'No statistics collected for the layer {edge_name}')
        return self._apply(model, engine, statistic_points)

    @abstractmethod
    def _apply(self, model: ModelType, engine: Engine, statistic_points: StatisticPointsContainer) -> ModelType:
        """
        Applies the algorithm to the model.
        """

    @abstractmethod
    def get_statistic_points(self, model: ModelType) -> StatisticPointsContainer:
        """
        Returns activation layers, for which StatisticsCollector should collect statistics.
        """


class CompositeAlgorithm(Algorithm):
    """
    Sub-class for comples Post-Training algorithms that contains other algorithms inside.
    """
    def __init__(self) -> None:
        super().__init__()
        self.algorithms = []

    @property
    def available_backends(self) -> Dict[str, BackendType]:
        # TODO(KodiaqQ): Need to add the implementation of the method (cross-algorithm backend calculation)
        # after updating the MinMax Quantization with shareable logic
        pass

    @abstractmethod
    def create_subalgorithms(self) -> None:
        """
        Some composite algorithms have different inner algorithms.
        This method creates sub-algorithms and sets model transformer to them
        """
