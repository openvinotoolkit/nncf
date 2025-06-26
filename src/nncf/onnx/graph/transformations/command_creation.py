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

import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.command_creation import CommandCreator
from nncf.common.graph.transformations.commands import TargetType
from nncf.onnx.graph.transformations.commands import ONNXInitializerUpdateCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint


def create_bias_correction_command(node: NNCFNode, bias_value: np.ndarray) -> ONNXInitializerUpdateCommand:
    """
     Creates bias correction command.

    :param node: The node in the NNCF graph that corresponds to operation with bias.
    :param bias_value: The new bias value that will be set.
    :return: The `ONNXInitializerUpdateCommand` command to update bias.
    """
    bias_port_id = node.metatype.bias_port_id
    target_point = ONNXTargetPoint(TargetType.LAYER, node.node_name, bias_port_id)
    return ONNXInitializerUpdateCommand(target_point, bias_value)


class ONNXCommandCreator(CommandCreator):
    """
    Implementation of the `CommandCreator` class for the ONNX backend.
    """

    @staticmethod
    def create_command_to_remove_quantizer(quantizer_node: NNCFNode) -> ONNXQDQNodeRemovingCommand:
        target_point = ONNXTargetPoint(TargetType.LAYER, quantizer_node.node_name, port_id=None)
        return ONNXQDQNodeRemovingCommand(target_point)

    @staticmethod
    def create_command_to_update_bias(
        node_with_bias: NNCFNode, bias_value: np.ndarray, nncf_graph: NNCFGraph
    ) -> ONNXInitializerUpdateCommand:
        return create_bias_correction_command(node_with_bias, bias_value)

    @staticmethod
    def create_command_to_update_weight(
        node_with_weight: NNCFNode, weight_value: np.ndarray, weight_port_id: int
    ) -> ONNXInitializerUpdateCommand:
        target_point = ONNXTargetPoint(TargetType.LAYER, node_with_weight.node_name, weight_port_id)
        return ONNXInitializerUpdateCommand(target_point, weight_value)

    @staticmethod
    def create_command_to_insert_bias(node_without_bias, bias_value):
        raise NotImplementedError
