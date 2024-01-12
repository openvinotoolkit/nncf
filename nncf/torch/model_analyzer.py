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

from typing import Optional

import torch

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.torch.graph.operator_metatypes import OP_NAMES_QUANTIZE_NODE
from nncf.torch.graph.operator_metatypes import OPERATORS_FUSED_METATYPES
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_BIAS_METATYPES
from nncf.torch.nncf_network import NNCFNetwork


def get_potential_fused_node(node_name: str, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Get next node that can contain fused bias in runtime.

    :param node_name: The node name.
    :param nncf_graph: The NNCF graph.
    :return: The node that can be fused or None.
    """
    target_node = nncf_graph.get_node_by_name(node_name)

    if target_node.metatype in OPERATORS_WITH_BIAS_METATYPES:
        next_nodes = nncf_graph.get_next_nodes(target_node)
        for node in next_nodes:
            if node.metatype in OPERATORS_FUSED_METATYPES:
                return node
    return None


def is_node_with_fused_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Checks if the node has a fused bias.

    :param node: The node to check.
    :param nncf_graph: The NNCF graph.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    fused_node = get_potential_fused_node(node.node_name, nncf_graph)

    return node.metatype in OPERATORS_WITH_BIAS_METATYPES and (
        node.layer_attributes.with_bias if fused_node is None else fused_node.layer_attributes.with_bias
    )


def get_fused_bias_value(node: NNCFNode, model: NNCFNetwork) -> Optional[torch.Tensor]:
    """
    Returns the bias tensor for the node or potential fused node.

    :param node: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    nncf_graph = model.nncf.get_graph()
    fused_node = get_potential_fused_node(node.node_name, nncf_graph)
    target_node_name = fused_node.node_name if fused_node else node.node_name
    node_module = model.nncf.get_containing_module(target_node_name)
    if node_module.bias is None:
        return None
    return node_module.bias.data


def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Check that module have fake_quantizer for weight.

    :param node: The target node.
    :param nncf_graph: The NNCF graph.
    :return bool: return `True` if the node is quantized.
    """
    for prev_node in nncf_graph.get_previous_nodes(node):
        if prev_node.node_type in OP_NAMES_QUANTIZE_NODE:
            return True
    return False
