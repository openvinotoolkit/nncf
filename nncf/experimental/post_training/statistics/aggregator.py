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

from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.experimental.post_training.api.engine import Engine
from nncf.experimental.post_training.api.dataloader import DataLoader

TensorType = TypeVar('TensorType')
ModelType = TypeVar('ModelType')


class StatisticsAggregator(ABC):
    """
    Base class for statistics collection.
    """

    def __init__(self, engine: Engine, dataloader: DataLoader):
        self.engine = engine
        self.dataloader = dataloader
        self.is_calculate_metric = False
        self.layers_statistics = {}  # type: Dict[str, TensorStatisticCollectorBase]

    @abstractmethod
    def collect_statistics(self, model: ModelType) -> None:
        """
        Collects statistics for layers determined in self.layers_statistics.
        The statistics are stored in self.layers_statistics.
        """

    def register_layer_statistics(self, layer_statistics: Dict[str, TensorStatisticCollectorBase]):
        """
        Registered layer for statistics collection.
        """
        # TODO: potentially could be intersection in layers_to_collect_statistics
        self.layers_statistics = layer_statistics
