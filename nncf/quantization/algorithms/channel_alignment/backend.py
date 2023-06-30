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

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, TypeVar

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.registry import Registry

TModel = TypeVar("TModel")
OutputType = TypeVar("OutputType")
ALGO_BACKENDS = Registry("algo_backends")


@dataclass
class DimsDescriptor:
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
        :param port_id: id of the port for the statistics disctribution.
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
        pass

    @staticmethod
    @abstractmethod
    def get_statistic_collector(
        reduction_shape, q: float, num_samples: int, inplace: bool
    ) -> TensorStatisticCollectorBase:
        pass

    @staticmethod
    @abstractmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        pass

    @staticmethod
    @abstractmethod
    def create_bias_update_command(node_with_bias: NNCFNode, updated_value: np.ndarray, nncf_graph: NNCFGraph):
        pass

    @staticmethod
    @abstractmethod
    def create_weights_update_command(node_with_weights: NNCFNode, updated_value: np.array, weights_port_id: int):
        pass

    @staticmethod
    @abstractmethod
    def get_dims_descriptor(node: NNCFNode) -> DimsDescriptor:
        pass

    @staticmethod
    @abstractmethod
    def get_conv_layer_attributes(node: NNCFNode) -> Optional[ConvolutionLayerAttributes]:
        pass

    @staticmethod
    def insert_null_biases(model: TModel) -> TModel:
        pass


class StatedTensor:
    def __init__(self, value: np.ndarray):
        self._value = value
        self._mod_times = 0

    @property
    def val(self):
        return self._value

    @val.setter
    def val(self, value):
        if self._value is None and value is None:
            return
        self._mod_times += 1
        self._value = value

    def is_modified(self) -> bool:
        return self._mod_times > 0


class ConvParamsContainer:
    def __init__(self, conv_op, model, nncf_graph, backend_entity: ChannelAlignmentAlgoBackend):
        _, self._weights_port_id = backend_entity.get_weights_port_ids_for_node(conv_op)
        self.stated_weight = StatedTensor(backend_entity.get_weight_value(conv_op, model, self._weights_port_id))
        bias = None
        if backend_entity.is_node_with_bias(conv_op, nncf_graph):
            bias = backend_entity.get_bias_value(conv_op, model, nncf_graph)
        self.stated_bias = StatedTensor(bias)
        self._op = conv_op
        self._dims = backend_entity.get_dims_descriptor(conv_op)

    @property
    def weight(self):
        return self.stated_weight.val

    @weight.setter
    def weight(self, value):
        self.stated_weight.val = value

    @property
    def bias(self):
        return self.stated_bias.val

    @bias.setter
    def bias(self, value):
        self.stated_bias.val = value

    @property
    def op(self):
        return self._op

    @property
    def weight_port_id(self):
        return self._weights_port_id

    @property
    def dims(self) -> DimsDescriptor:
        return self._dims

    def has_bias(self) -> bool:
        return self.bias is not None
