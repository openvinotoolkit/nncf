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
from typing import List
from typing import Optional

from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.tensor_statistics.collectors import PercentageOfZerosStatisticCollector
from nncf.common.utils.registry import Registry

ALGO_BACKENDS = Registry('algo_backends')


class ActivationSparsityStatisticAlgoBackend(ABC):

    @staticmethod
    @abstractmethod
    def percentage_of_zeros_statistic_collector(
        num_samples: Optional[int] = None,
    ) -> PercentageOfZerosStatisticCollector:
        """
        Returns backend-specific instance of the NNCFCollectorTensorProcessor.

        :param num_samples:

        :return: Backend-specific NNCFCollectorTensorProcessor.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_node_name: Name of the located node.
        :param port_id: id of the port.

        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def default_target_node_types() -> List[str]:
        """
        Returns the list of node types for which statistics will be collected.

        :return: list of node types.
        """

    @staticmethod
    @abstractmethod
    def ignored_input_node_types() -> List[str]:
        """
        Return the list of node types that will be ignored as input node.

        :return: list of node types.
        """
