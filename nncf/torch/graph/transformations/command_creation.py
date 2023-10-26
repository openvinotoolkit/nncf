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

from typing import List

import torch
from torch import Tensor

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTWeightUpdateCommand


def create_bias_correction_command(node: NNCFNode, bias_value: Tensor) -> PTBiasCorrectionCommand:
    """
     Creates bias correction command.

    :param node: The node in the NNCF graph that corresponds to operation with bias.
    :param bias_value: The new bias value that will be set.
    :return: The `PTBiasCorrectionCommand` command to update bias.
    """
    target_point = PTTargetPoint(TargetType.LAYER, node.node_name)
    return PTBiasCorrectionCommand(target_point, bias_value)


def create_command_to_update_weight(node: NNCFNode, weight_value: Tensor) -> PTWeightUpdateCommand:
    """
     Creates weight update command.

    :param node: The node in the NNCF graph that corresponds to operation with weight.
    :param weight_value: The new weight value that will be set.
    :return: The `PTWeightUpdateCommand` command to update weight.
    """
    target_point = PTTargetPoint(TargetType.LAYER, node.node_name)
    return PTWeightUpdateCommand(target_point, weight_value)


class SQMultiply(torch.nn.Module):
    def __init__(self, scale_value):
        super().__init__()
        self._scale_value = scale_value

    def forward(self, x):
        return torch.mul(x, self._scale_value)


def multiply_insertion_command(
    target_nodes: List[NNCFNode], scale_value: Tensor, scale_node_name: str, input_port_id: int
) -> PTInsertionCommand:
    target_points = []
    for target_node in target_nodes:
        target_points.append(
            PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, target_node.node_name, input_port_id=input_port_id)
        )

    return PTSharedFnInsertionCommand(target_points, SQMultiply(scale_value), scale_node_name)
