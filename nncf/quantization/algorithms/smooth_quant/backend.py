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
from typing import Callable, List, Tuple, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.tensor import Tensor

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class SmoothQuantAlgoBackend(ABC):
    @property
    @abstractmethod
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        """
        Parameter for backend-specific metatypes for Convolution.

        :return: OperatorMetatype list.
        """

    @property
    @abstractmethod
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        """
        Parameter for backend-specific metatypes for MatMul.

        :return: OperatorMetatype list.
        """

    @property
    @abstractmethod
    def quantize_agnostic_metatypes(self) -> List[OperatorMetatype]:
        """
        Parameter for backend-specific quantize agnostic metatypes.

        :return: List of OperatorMetatype.
        """

    @staticmethod
    @abstractmethod
    def pre_layer_target_type() -> TargetType:
        """
        Returns backend-specific pre layer target type.

        :returns: Backend-specific pre layer target type.
        """

    @staticmethod
    @abstractmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> TargetPoint:
        """
        Returns backend-specific target point.

        :param target_type: Type of the location that should be modified.
        :param target_node_name: Name of the located node.
        :param port_id: Port ID of the tensor for the statistics distribution.
        :return: Backend-specific TargetPoint.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        """
        Checks whether the node with weights or not.

        :param node: NNCFNode to check.
        :return: Boolean indicating whether the node has weights or not.
        """

    @staticmethod
    @abstractmethod
    def get_activations_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        """
        Returns map with activation & weighted ports.

        :param node: NNCFNode to check.
        :param nncf_graph: NNCFGraph instance.
        :return: Map with the activation & weighted ports.
        """

    @staticmethod
    @abstractmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: Tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        """
        Returns TensorCollector with MaxAggregator and AbsMaxReducer.

        :param stats_reduction_axes: Calculated reduction axes.
        :param inplace: Whether to calculate statistic inplace or not.
        :param branch_key: Specific string for branch key.
        :return: TensorCollector instance.
        """

    @staticmethod
    @abstractmethod
    def get_weight_value(node_with_weight: NNCFNode, model: TModel, port_id: int, nncf_graph: NNCFGraph) -> Tensor:
        """
        Returns the weight value for the node with weight.

        :param node_with_weight: The node with weight.
        :param model: The model that contains this operation.
        :param port_id: The input port ID to get weight input.
        :param nncf_graph: NNCFGraph instance.
        :return: The weight value.
        """

    @staticmethod
    @abstractmethod
    def weight_update_command(
        node_with_weight: NNCFNode, weight_value: TTensor, weight_port_id: int
    ) -> TransformationCommand:
        """
        Returns command to update weights.

        :param node_with_weight: NNCFNode instance.
        :param weight_value: New weight value.
        :param weight_port_id: Weight port id.
        :return: TransformationCommand instance.
        """

    @staticmethod
    @abstractmethod
    def scale_insertion_command(
        source_node: NNCFNode, scale_value: TTensor, source_output_port_id: int, nodes: List[NNCFNode]
    ) -> TransformationCommand:
        """
        Returns command to insert Smooth Quant node.

        :param source_node: NNCFNode instance.
        :param scale_value: Smooth Quant value.
        :param source_output_port_id: Output port for source node.
        :param nodes: List of consumers for Smooth node.
        :return: TransformationCommand instance.
        """

    @staticmethod
    @abstractmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        """
        Returns axis number of the activation tensor which correspond to it channel.

        :param node: NNCFNode instance.
        :param port_id: Specified input port id.
        :return: Channel axis number.
        """

    @staticmethod
    @abstractmethod
    def get_weight_channel_axis(node: NNCFNode) -> int:
        """
        Returns axis number of the weight tensor which correspond to it channel.

        :param node: NNCFNode instance.
        :return: Channel axis number.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_shared_weight(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Returns true if given node shares constant with a different node.

        :param node: NNCFNode instance.
        :param nncf_graph: NNCFGraph instance.
        :return: Whether the given node is shares weights with a different node or not.
        """

    @staticmethod
    @abstractmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        """
        Returns backend-specific callable to filter statistic containers according to its statistic point.

        :param activation_port_id: Activation port id for the statistic collection target node.
        :param algorithm_key: Current algorithm key.
        :return: Backend-specific callable to filter statistic containers according to its statistic point.
        """
