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
from typing import Any

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TransformationCommand


class CommandCreator(ABC):
    """
    Creates transformation commands to use them in the model transformer.
    """

    @staticmethod
    @abstractmethod
    def create_command_to_remove_quantizer(quantizer_node: NNCFNode) -> TransformationCommand:
        """
        Creates command to remove quantizer from the quantized model.

        :param quantizer_node: The quantizer which should be removed.
        :return: The command to remove quantizer from the quantized model.
        """

    @staticmethod
    @abstractmethod
    def create_command_to_update_bias(
        node_with_bias: NNCFNode, bias_value: Any, nncf_graph: NNCFGraph
    ) -> TransformationCommand:
        """
        Creates command to update bias value.

        :param node_with_bias: The node that corresponds to the operation with bias.
        :param bias_value: New bias value.
        :param nncf_graph: The NNCF graph.
        :return: The command to update bias value.
        """

    @staticmethod
    @abstractmethod
    def create_command_to_insert_bias(node_without_bias: NNCFNode, bias_value: Any) -> TransformationCommand:
        """
        Creates command to insert bias after given node.

        :param node_without_bias: The node that corresponds to the operation without bias.
        :param bias_value: Bias value to insert.
        :return: The command to insert bias value.
        """

    @staticmethod
    @abstractmethod
    def create_command_to_update_weight(
        node_with_weight: NNCFNode, weight_value: Any, weight_port_id: int
    ) -> TransformationCommand:
        """
        Creates command to update weight value.

        :param node_with_weight: The node that corresponds to the operation with weight.
        :param weight_value: New weight value.
        :param weight_port_id: The input port ID that corresponds to weight.
        :return: The command to update weight value.
        """
