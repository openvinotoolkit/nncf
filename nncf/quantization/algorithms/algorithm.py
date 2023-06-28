# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from typing import Dict, Optional, TypeVar

from nncf import Dataset
from nncf.common.tensor_statistics.statistic_point import StatisticPointsContainer
from nncf.common.utils.backend import BackendType

TModel = TypeVar("TModel")


class AlgorithmParameters(ABC):
    """
    Base class for Post-Training algorithm parameters.
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

    def apply(
        self,
        model: TModel,
        statistic_points: Optional[StatisticPointsContainer] = None,
        dataset: Optional[Dataset] = None,
    ) -> TModel:
        """
        Checks that statistic point exists, sets model into transformer
        and applies the algorithm to the model.
        :param model: model for applying algorithm
        :param engine: engine for the model execution
        :param statistic_points: StatisticPointsContainer
        :return: model after algorithm
        """
        # TODO (asuslov): add validation statistic_points
        return self._apply(model, statistic_points=statistic_points, dataset=dataset)

    @abstractmethod
    def _apply(
        self, model: TModel, statistic_points: StatisticPointsContainer, dataset: Optional[Dataset] = None
    ) -> TModel:
        """
        Applies the algorithm to the model.
        """

    @abstractmethod
    def get_statistic_points(self, model: TModel) -> StatisticPointsContainer:
        """
        Returns activation layers, for which StatisticsCollector should collect statistics.
        """
