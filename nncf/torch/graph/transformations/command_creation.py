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

from typing import List, Optional, Union

import torch
from torch import Tensor

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.torch.graph.transformations.commands import ExtraCompressionModuleType
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTWeightUpdateCommand
from nncf.torch.quantization.layers import BaseQuantizer


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


def create_quantizer_insertion_command(
    target_point: PTTargetPoint, quantizer: BaseQuantizer
) -> Union[PTInsertionCommand, PTSharedFnInsertionCommand]:
    quantizer_id = NonWeightQuantizerId(target_point.target_node_name, target_point.input_port_id)
    storage_key = str(quantizer_id)
    return PTSharedFnInsertionCommand(
        target_points=[target_point],
        fn=quantizer,
        op_unique_name=storage_key,
        compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
        priority=TransformationPriority.QUANTIZATION_PRIORITY,
    )


def create_shared_quantizer_insertion_command(
    target_points: List[PTTargetPoint], quantizer: BaseQuantizer
) -> PTSharedFnInsertionCommand:
    quantizers_ids = []
    for target_point in target_points:
        quantizers_ids.append(NonWeightQuantizerId(target_point.target_node_name, target_point.input_port_id))

    storage_key = ";".join(str(quantizer_id) for quantizer_id in sorted(quantizers_ids, key=str))
    return PTSharedFnInsertionCommand(
        target_points=target_points,
        fn=quantizer,
        op_unique_name=storage_key,
        compression_module_type=ExtraCompressionModuleType.EXTERNAL_QUANTIZER,
        priority=TransformationPriority.QUANTIZATION_PRIORITY,
    )


def create_pt_insertion_command(
    module: torch.nn.Module,
    target_type: TargetType,
    target_node_name: str,
    priority: int,
    input_port_id: Optional[int],
) -> PTInsertionCommand:
    """
    Creates a PTInsertionCommand.

    :param module: Torch module to insert.
    :param target_type: Insertion command target type.
    :param target_name: Insertion command target name.
    :param priority: Insertion command priority.
    :param input_port_id: Insertion command input port id.
    :return: A PTInsertionCommand
    """
    target_point = PTTargetPoint(
        target_type=target_type, target_node_name=target_node_name, input_port_id=input_port_id
    )
    return PTInsertionCommand(point=target_point, fn=module, priority=priority)
