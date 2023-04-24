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
from typing import Optional

from torch import Tensor

from nncf.common.graph.graph import NNCFNode
from nncf.common.logging import nncf_logger
from nncf.torch.graph.operator_metatypes import OPERATORS_FUSED_METATYPES
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_BIAS_METATYPES
from nncf.torch.layer_utils import _NNCFModuleMixin
from nncf.torch.nncf_network import NNCFNetwork


def get_next_fused_bias_node(node_name: NNCFNode, model: NNCFNetwork) -> Optional[NNCFNode]:
    """
    Get next node that can contain fused bias in runtime.
    Available only for Convolutional and MatMul layers.

    :param node_name: The node name.
    :param model: The model that contains this operation.

    :return NNCFNode: The node that can be fused or None.
    """
    graph = model.nncf.get_original_graph()
    target_node = graph.get_node_by_name(node_name)

    if target_node.metatype in OPERATORS_WITH_BIAS_METATYPES:
        next_nodes = graph.get_next_nodes(target_node)
        for node in next_nodes:
            if node.metatype in OPERATORS_FUSED_METATYPES:
                node_module = model.get_containing_module(node.node_name)
                if not isinstance(node_module, _NNCFModuleMixin):
                    # Disable detect fused bias in custom batch_norm modules, like BatchNormAct2d from timm.
                    nncf_logger.debug(
                        f"Module {node.node_name}({type(node_module)}) is not inherited from _NNCFModuleMixin"
                    )
                    return None
                return node
    return None


def is_node_with_fused_bias(node: NNCFNode, model: NNCFNetwork) -> bool:
    """
    Checks if the node has a fused bias.

    :param node: The node to check.
    :param model: The model that contains this operation.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    next_norm_node = get_next_fused_bias_node(node.node_name, model)
    node_module = model.get_containing_module(node.node_name)

    return node.metatype in OPERATORS_WITH_BIAS_METATYPES and (
        node_module.bias is not None or next_norm_node is not None
    )


def get_fused_bias_value(node: NNCFNode, model: NNCFNetwork) -> Tensor:
    """
    Returns the fused bias tensor for the biased node.

    :param node: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    next_norm_node = get_next_fused_bias_node(node.node_name, model)
    target_node_name = next_norm_node.node_name if next_norm_node else node.node_name
    node_module = model.get_containing_module(target_node_name)
    if node_module.bias is None:
        return None
    return node_module.bias.data
