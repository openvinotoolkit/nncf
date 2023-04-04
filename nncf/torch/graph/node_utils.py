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
from torch import Tensor
from nncf.common.graph.graph import NNCFNode
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.graph.operator_metatypes import OPERATORS_WITH_BIAS_METATYPES


def is_node_with_bias(node: NNCFNode) -> bool:
    """
    Checks if the node has a bias or not.

    :param node: The node to check.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    return node.metatype in OPERATORS_WITH_BIAS_METATYPES


def get_bias_value(node: NNCFNode, model: NNCFNetwork) -> Tensor:
    """
    Returns the bias tensor for the biased node.

    :param node: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    node_module = model.get_containing_module(node.node_name)
    if node_module.bias is None:
        return None
    return node_module.bias.data
