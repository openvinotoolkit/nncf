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
from typing import Dict, List, Optional, Tuple, TypeVar

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationCommand
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class SmoothQuantAlgoBackend(ABC):
    @property
    @abstractmethod
    def convolution_metatype(self) -> OperatorMetatype:
        """
        Parameter for backend-specific metatype for Convolution.

        :return: OperatorMetatype
        """

    @property
    @abstractmethod
    def matmul_metatype(self) -> OperatorMetatype:
        """
        Parameter for backend-specific metatype for MatMul.

        :return: OperatorMetatype
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
    def get_input_ports_map(node: NNCFNode, nncf_graph: NNCFGraph) -> Dict[str, int]:
        """
        Returns map with activation & weighted ports.

        :param node: NNCFNode to check.
        :param nncf_graph: NNCFGraph instance.
        :return: Map with the activation & weighted ports.
        """

    @staticmethod
    @abstractmethod
    def get_channel_agnostic_reduction_axes(channel_axis: int, shape: Tuple[int]) -> Tuple[int]:
        """
        Returns filtered reduction axes without axes that corresponds channels.

        :param channel_axes: List of the channel axes.
        :param shape: Shape that need to be filtered.
        :return: Reduction axes in tuple format.
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
    def process_weight_statistics(weights: TTensor, channel_axis: int) -> TTensor:
        """
        Returns processed weight statistics for node.

        :param weights: Weights tensor.
        :param channel_axis: Channel axis for calculation.
        :return: Weight statistics.
        """

    @staticmethod
    @abstractmethod
    def get_weight_value(node_with_weight: NNCFNode, model: TModel, port_id: int) -> TTensor:
        """
        Returns the weight value for the node with weight.

        :param node_with_weight: The node with weight.
        :param model: The model that contains this operation.
        :param port_id: The input port ID to get weight input.
        :return: The weight value.
        """

    @staticmethod
    @abstractmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        """
        Returns node's input port indices with weights tensors.

        :param node: NNCFNode to find its weights input port indices.
        :return: Weights input port indices.
        """

    @staticmethod
    @abstractmethod
    def clip_statistics(statistics: TTensor) -> TTensor:
        """
        Clips statistics for further calculation.

        :param statistics: Input statistics.
        :return: Clipped statistics.
        """

    @staticmethod
    @abstractmethod
    def calculate_scale_and_ratio(
        activations: TTensor, weights: TTensor, alpha: float, quantile: Optional[float]
    ) -> Tuple[TTensor, TTensor]:
        """
        Calculates base scale value and it's ratio.

        :param activations: Activation statistics value.
        :param weights: Weights statistics value.
        :param alpha: Base value for exponentiation.
        :param quantile: Base quantile value.
        :return: Calculated base scale value & ratio.
        """

    @staticmethod
    @abstractmethod
    def calculate_activation_scale(scale_value: TTensor, activations_size: int, channel_axis: int) -> TTensor:
        """
        Calculates activation scales for Smooth node.

        :param scale_value: Base scale value.
        :param activations_size: Size of the activation shape.
        :param channel_axis: Axis for shape calculation.
        :return: Calculated activation scale.
        """

    @staticmethod
    @abstractmethod
    def calculate_weight_scale(scale_value: TTensor, weights_size: int, channel_axis: int) -> TTensor:
        """
        Calculates scale for weight tensor.

        :param scale_value: Base scale value.
        :param weights_size: Size of the weights shape.
        :param channel_axis: Axis for shape calculation.
        :return: Calculated scale for weights.
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
        source_node: NNCFNode, scale_value: TTensor, port_id: int, nodes: List[NNCFNode]
    ) -> TransformationCommand:
        """
        Returns command to insert Smooth Quant node.

        :param source_node: NNCFNode instance.
        :param scale_value: Smooth Quant value.
        :param port_id: Output port for source node.
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
    def get_weight_channel_axis(node: NNCFNode, port_id: int) -> int:
        """
        Returns axis number of the weight tensor which correspond to it channel.

        :param node: NNCFNode instance.
        :param port_id: Specified input port id.
        :return: Channel axis number.
        """

    @staticmethod
    @abstractmethod
    def calculate_port_based_channel_axis(port_id: int, transpose: bool) -> int:
        """
        Returns port-based channel axis.

        :param port_id: Specified input port id.
        :param transpose: Transpose position.
        :return: Channel axis.
        """
