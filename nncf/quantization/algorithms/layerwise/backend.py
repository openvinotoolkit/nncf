# Copyright (c) 2025 Intel Corporation
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
from typing import Dict, List, Optional, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.layerwise.iterator import LayerwiseIterator
from nncf.quantization.algorithms.layerwise.scheduler import LayerwiseStep
from nncf.quantization.algorithms.layerwise.scheduler import NodeOutputPort
from nncf.tensor import Tensor

TModel = TypeVar("TModel")


class LayerwiseEngineBackend(ABC):
    @staticmethod
    @abstractmethod
    def create_layerwise_iterator(
        model: TModel,
        graph: NNCFGraph,
        schedule: List[LayerwiseStep],
        dataset: Dataset,
        subset_size: int = 100,
        cache: Optional[Dict[NodeOutputPort, List[Tensor]]] = None,
    ) -> LayerwiseIterator:
        """
        Creates an iterator to iterate through layers according to a given schedule.

        This method generates an iterator that processes each layer of the model sequentially,
        following the specified schedule. It allows for detailed examination and manipulation
        of the model's intermediate layer outputs.

        :param model: The model to be iterated over.
        :param graph: The model graph.
        :param schedule: A list of steps defining how each layer should be processed.
        :param dataset: The dataset to be used for obtaining inputs to the model.
        :param subset_size: The number of samples from the dataset to use, defaults to 100.
        :param cache: Optional cache to store and reuse layer outputs, defaults to None.
        :return: An iterator yielding specified inputs or outputs for each layer of the model.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: id of the port for the statistics distribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorStatisticCollectorBase:
        """
        Returns backend-specific raw statistic collector.
        This statistic collector is used for raw data calculation, without aggregating.

        :param num_samples: Maximum number of samples to collect.
        :return: Backend-specific TensorStatisticCollectorBase for the statistics calculation.
        """
