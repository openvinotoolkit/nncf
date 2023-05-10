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

import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.command_creation import CommandCreator
from nncf.common.graph.transformations.commands import TargetType
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.openvino.graph.transformations.commands import OVNullBiasInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand


class OVCommandCreator(CommandCreator):
    """
    Implementation of the `CommandCreator` class for the OpenVINO backend.
    """

    @staticmethod
    def create_command_to_remove_quantizer(quantizer_node: NNCFNode) -> OVFQNodeRemovingCommand:
        target_point = OVTargetPoint(TargetType.LAYER, quantizer_node.node_name, port_id=None)
        return OVFQNodeRemovingCommand(target_point)

    @staticmethod
    def create_command_to_update_bias(
        node_with_bias: NNCFNode, bias_value: np.ndarray, nncf_graph: NNCFGraph
    ) -> OVBiasCorrectionCommand:
        add_node = nncf_graph.get_next_nodes(node_with_bias)[0]
        const_port_ids = add_node.layer_attributes.get_const_port_ids()
        assert len(const_port_ids) == 1
        bias_port_id = const_port_ids[0]
        target_point = OVTargetPoint(TargetType.LAYER, node_with_bias.node_name, bias_port_id)
        return OVBiasCorrectionCommand(target_point, bias_value)

    @staticmethod
    def create_command_to_update_weight(
        node_with_weight: NNCFNode, weight_value: np.ndarray, weight_port_id: int
    ) -> OVWeightUpdateCommand:
        target_point = OVTargetPoint(TargetType.LAYER, node_with_weight.node_name, weight_port_id)
        return OVWeightUpdateCommand(target_point, weight_value)

    @staticmethod
    def create_command_to_insert_bias(node_without_bias: NNCFNode) -> OVNullBiasInsertionCommand:
        target_point = OVTargetPoint(TargetType.POST_LAYER_OPERATION, node_without_bias.node_name, 0)
        return OVNullBiasInsertionCommand(target_point)
