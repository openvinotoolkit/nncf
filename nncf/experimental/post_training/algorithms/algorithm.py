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
from nncf.experimental.post_training.graph.model_transformer import StaticModelTransformerBase
from nncf.experimental.post_training.statistics.statistic_point import StatisticPointsContainer
from nncf.experimental.post_training.api.engine import Engine
from nncf.common.utils.backend import BackendType

ModelType = TypeVar('ModelType')


class PostTrainingAlgorithms(Enum):
    MinMaxQuantization = 'min_max_quantization'
    FastBiasCorrection = 'fast_bias_correction'
    PostTrainingQuantization = 'post_training_quantization'


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

    def __init__(self) -> None:
        self._model_transformer = None

    @property
    def model_transformer(self) -> StaticModelTransformerBase:
        if self._model_transformer is None:
            raise RuntimeError('model_transformer variable was not set before call')
        return self._model_transformer

    @model_transformer.setter
    def model_transformer(self, model_transformer: StaticModelTransformerBase) -> None:
        self._model_transformer = model_transformer

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
        self.model_transformer.set_model(model)
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

    def create_subalgorithms(self, backend: BackendType) -> None:
        """
        Some composite algorithms have different inner algorithms.
        This method creates sub-algorithms and sets model transformer to them

        :param backend: backend for the algorithms differentiation
        """
        self._create_subalgorithms(backend)
        for algorithm in self.algorithms:
            algorithm.model_transformer = self.model_transformer

    @abstractmethod
    def _create_subalgorithms(self, backend: BackendType) -> None:
        """
        Some composite algorithms have different inner algorithms.
        This method creates sub-algorithms
        """
