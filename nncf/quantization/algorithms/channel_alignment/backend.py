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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Tuple, TypeVar

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase

TModel = TypeVar("TModel")


@dataclass
class LayoutDescriptor:
    """
    Container to store convolutional and linear layers layout information.
    """

    conv_weight_out_channels_dim: int
    conv_weight_in_channels_dim: int
    bias_channels_dim: int


class ChannelAlignmentAlgoBackend:
    @staticmethod
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
    def get_bias_value(node: NNCFNode, model: TModel, nncf_graph: NNCFGraph) -> np.ndarray:
        """
        Returns bias value in the NumPy format of provided node.

        :param node: Node of NNCFGraph with bias value.
        :param model: Backend-specific model for the initializer finding.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Bias value in the NumPy format.
        """

    @staticmethod
    @abstractmethod
    def get_weight_value(node: NNCFNode, model: TModel, port_id: int) -> np.ndarray:
        """
        Returns bias value in the NumPy format of provided node.

        :param node: Node of NNCFGraph with bias value.
        :param model: Backend-specific model for the initializer finding.
        :param nncf_graph: NNCFGraph instance with the node.
        :return: Bias value in the NumPy format.
        """

    @staticmethod
    @abstractmethod
    def get_activation_port_ids_for_node(node: NNCFNode) -> Tuple[int, int]:
        """
        Returns Input Port ID and Output Port ID corresponding to activation input and output edges for
        the node.
        Supports only nodes that could have bias value.

        :param node: Node of NNCFGraph with bias value.
        """

    @staticmethod
    @abstractmethod
    def get_weights_port_ids_for_node(node: NNCFNode) -> Tuple[int, int]:
        """
        Returns Input Port ID and Output Port ID corresponding to node weights input port id and
        constant output port id the node.

        :param node: Node of NNCFGraph.
        """

    @staticmethod
    @abstractmethod
    def get_statistic_collector(
        reduction_axes, q: float, num_samples: int, inplace: bool
    ) -> TensorStatisticCollectorBase:
        """
        Get backend-specific tensor collector that collects medians of minimal and maximal quantiles.

        :param reduction_axes: Target reduction axes for the reduction.
        :param q: Minimal quantile for the tensor collector.
        :param num_samples: Num samples to collect by the tensor collector.
        :param inplace: Should statistic be calculated inplace or out of place.
        :return: Backend-specific tensor collector that collects medians of minimal and maximal quantiles.
        """

    @staticmethod
    @abstractmethod
    def get_conv_layer_attributes(node: NNCFNode) -> ConvolutionLayerAttributes:
        """
        Returns convolutional layer attributes of given node if they are present and None otherwise.
        :param node: NNCFNode to take convolutional layer attributes from.
        :return: Convolutional layer attributes of given node if they are present and None otherwise
        """

    @staticmethod
    @abstractmethod
    def get_dims_descriptor(node: NNCFNode) -> LayoutDescriptor:
        """
        Return weights layout descriptor of the given node if it is possible and None otherwise.
        Only convolutional and linear nodes are supported.

        :param node: NNCFNode to get layout descriptor from.
        :return: Weights layout descriptor of the given node if it is possible and None otherwise.
        """

    @staticmethod
    @abstractmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        """
        Checks if the node has a bias or not.

        :param node: The node to check.
        :param nncf_graph: The NNCF graph.
        :return: True` if `node` corresponds to the operation with bias
            (bias is added to the output tensor of that operation), `False` otherwise.
        """

    @staticmethod
    @abstractmethod
    def create_bias_tensor(node: NNCFNode, nncf_graph: NNCFGraph, value: Any) -> np.ndarray:
        """
        Creates bias value constant array filled by given value.

        :param node: NNCFNode to add bias to.
        :param nncf_graph: Target NNCFgraph.
        :param value: Value to fill bias constant array.
        :return: Bias value constant array filled by given value.
        """
