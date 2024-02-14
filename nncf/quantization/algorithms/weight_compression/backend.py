# Copyright (c) 2024 Intel Corporation
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
from typing import Dict, Iterable, List, Optional, Tuple, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.experimental.tensor import Tensor
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionParameters

TModel = TypeVar("TModel")


class WeightCompressionAlgoBackend(ABC):
    @property
    @abstractmethod
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for matmul layers.
        """

    @property
    @abstractmethod
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for convolution layers.
        """

    @property
    @abstractmethod
    def embedding_metatypes(self) -> List[OperatorMetatype]:
        """
        Property for the backend-specific metatypes for embedding layers.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_weights(node: NNCFNode, graph: NNCFGraph) -> bool:
        """
        Checks whether the node with weights or not.

        :param node: The node to check.
        :param graph: The model graph.
        :return: True if the node contains weights, False otherwise.
        """

    @staticmethod
    @abstractmethod
    def get_reduction_axes(node_with_weight: NNCFNode, weight_port_id: int, graph: NNCFGraph) -> Optional[Tuple[int]]:
        """
        Returns reduction axes without axes that corresponds to weight channels of the node with weight.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The input port ID that corresponds to weight.
        :param graph: The model graph.
        :return: Reduction shape in tuple format or None if not applicable.
        """

    @staticmethod
    @abstractmethod
    def get_weight_names_and_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Tuple[str, int]]:
        """
        Returns a list of weight names and port ids for the given node.

        :param node: The node.
        :param graph: The model graph.
        :return: List of tuples with weight names and port ids.
        """

    @abstractmethod
    def get_weight(self, node_with_weight: NNCFNode, weight_port_id: int, model: TModel, graph: NNCFGraph) -> Tensor:
        """
        Returns a weight associated with the given node on the given port id.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The weight port id for given node with weight.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :return: The weight tensor.
        """

    @abstractmethod
    def set_weight(
        self, node_with_weight: NNCFNode, weight_port_id: int, model: TModel, graph: NNCFGraph, weight: Tensor
    ) -> None:
        """
        Update a weight associated with the given node on the given port id.

        :param node_with_weight: The node with weight.
        :param weight_port_id: The weight port id for given node with weight.
        :param model: The model.
        :param graph: The model graph associated with the model.
        :param weight: The weight tensor.
        """

    @abstractmethod
    def transform_model(
        self, model: TModel, graph: NNCFGraph, weight_compression_parameters: Iterable[WeightCompressionParameters]
    ) -> TModel:
        """
        Applies weight compression transformations to the model.

        :param model: Model in which the weights will be compressed according to the weight compression description.
        :param graph: The graph associated with the model.
        :param weight_compression_parameters: List of weight compression parameters.
        :return: The transformed model.
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

    @staticmethod
    @abstractmethod
    def get_activation_port_id(node: NNCFNode, graph: NNCFGraph) -> int:
        """
        Returns input port id corresponding to activation input edge for the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        :param graph: NNCFGraph instance with the node.
        :return: target input port id.
        """

    @staticmethod
    def dump_parameters(
        model: TModel, parameters: Dict, algo_name: Optional[str] = "quantization", path: Optional[List] = None
    ) -> None:
        """
        Dumps the given parameters into Model's meta section.

        :param model: ov.Model instance.
        :param algo_name: Name of the algorithm to which the parameters refer.
        :param parameters: Incoming dictionary with parameters to save.
        :param path: Optional list of the paths.
        """


class AWQAlgoBackend(WeightCompressionAlgoBackend):
    @staticmethod
    def get_awq_patterns() -> Dict:
        """
        Returns patterns of nodes in network graph for applying AWQ algorithm.
        """
