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

from typing import List

from abc import ABC
from abc import abstractmethod

from nncf.experimental.post_training.statistics.statistics_collector import LayerStatistic
from nncf.experimental.post_training.algorithms import Algorithm
from nncf.experimental.post_training.algorithms import AlgorithmParameters


class BiasCorrectionAlgorithmParameters(AlgorithmParameters):
    pass


class BiasCorrectionAlgorithm(Algorithm, ABC):
    @abstractmethod
    def get_layers_for_statistics(self) -> List[LayerStatistic]:
        pass

    @abstractmethod
    def get_transformation_commands(self, layers_statistics):
        pass
