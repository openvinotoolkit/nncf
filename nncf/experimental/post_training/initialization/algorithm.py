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

from typing import Dict
from typing import Callable

from abc import ABC
from abc import abstractmethod

from nncf.experimental.post_training.compressed_model import CompressedModel
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.initialization.statistics_collector import StatisticsCollector
from nncf.common.graph.transformations.layout import TransformationCommand


class InitializationAlgorithm(ABC):
    """
    The base class for all post-training quantization initialization algorithms.
    """

    def __init__(self, compressed_model: CompressedModel, engine: Engine, **kwargs):
        self.compressed_model = compressed_model
        self.engine = engine

    @abstractmethod
    def get_layers_for_statistics(self) -> Dict[str, Callable]:
        """

        """

    @abstractmethod
    def get_transformation_commands(self, collector: StatisticsCollector) -> TransformationCommand:
        """

        """
