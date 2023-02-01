"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.graph.transformations.commands import OVBiasCorrectionCommand


def create_bias_correction_command(node: NNCFNode,
                                   bias_value: np.ndarray,
                                   nncf_graph: NNCFGraph) -> OVBiasCorrectionCommand:
    """
    Creates bias correction command.

    :param node: The node in the NNCF graph that corresponds to operation with bias.
    :param bias_value: The new bias value that will be set.
    :param nncf_graph: NNCFGraph instance that contains the node.
    :return: The `OVBiasCorrectionCommand` command to update bias.
    """
    add_node = nncf_graph.get_next_nodes(node)[0]
    bias_port_id = add_node.layer_attributes.const_port_id
    target_point = OVTargetPoint(TargetType.LAYER, node.node_name, bias_port_id)
    return OVBiasCorrectionCommand(target_point, bias_value)
