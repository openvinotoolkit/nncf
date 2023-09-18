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

from torch import Tensor

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint


def create_bias_correction_command(node: NNCFNode, bias_value: Tensor) -> PTBiasCorrectionCommand:
    """
     Creates bias correction command.

    :param node: The node in the NNCF graph that corresponds to operation with bias.
    :param bias_value: The new bias value that will be set.
    :return: The `PTBiasCorrectionCommand` command to update bias.
    """
    target_point = PTTargetPoint(TargetType.LAYER, node.node_name)
    return PTBiasCorrectionCommand(target_point, bias_value)
