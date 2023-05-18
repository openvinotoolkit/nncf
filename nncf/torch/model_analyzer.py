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
from torch import nn
from torch.quantization.fake_quantize import FakeQuantize

from nncf.common.graph.graph import NNCFNode
from nncf.torch.graph.operator_metatypes import OPERATORS_FUSED_METATYPES
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_BIAS_METATYPES
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer


def get_potential_fused_node(node_name: str, model: NNCFNetwork) -> Optional[NNCFNode]:
    """
    Get next node that can contain fused bias in runtime.

    :param node_name: The node name.
    :param model: The model that contains this operation.

    :return: The node that can be fused or None.
    """
    graph = model.nncf.get_original_graph()
    target_node = graph.get_node_by_name(node_name)

    if target_node.metatype in OPERATORS_WITH_BIAS_METATYPES:
        next_nodes = graph.get_next_nodes(target_node)
        for node in next_nodes:
            if node.metatype in OPERATORS_FUSED_METATYPES:
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
    fused_node = get_potential_fused_node(node.node_name, model)
    node_module = model.nncf.get_containing_module(node.node_name)

    return node.metatype in OPERATORS_WITH_BIAS_METATYPES and (node_module.bias is not None or fused_node is not None)


def get_fused_bias_value(node: NNCFNode, model: NNCFNetwork) -> Optional[torch.Tensor]:
    """
    Returns the bias tensor for the node or potential fused node.

    :param node: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    fused_node = get_potential_fused_node(node.node_name, model)
    target_node_name = fused_node.node_name if fused_node else node.node_name
    node_module = model.get_containing_module(target_node_name)
    if node_module.bias is None:
        return None
    return node_module.bias.data


def find_fake_quantizer_for_weight(module: nn.Module) -> Optional[nn.Module]:
    """
    Return quantizer operator for weight of module if exists, otherwise return None.

    :param module: The target module.

    :return nn.Module: Quantizer module.
    """
    for pre_op in module.pre_ops.values():
        if isinstance(pre_op, UpdateWeight) and isinstance(pre_op.op, (BaseQuantizer, FakeQuantize)):
            return pre_op.op
    return None


def is_quantized_weights(node: NNCFNode, model: NNCFNetwork) -> bool:
    """
    Check that module have fake_quantizer for weight.

    :param node: The target node.
    :param model: The model.

    :return bool: return `True` if module have FQ pre_ops for weight.
    """
    node_module = model.nncf.get_containing_module(node.node_name)
    fq_module = find_fake_quantizer_for_weight(node_module)
    return fq_module is not None
