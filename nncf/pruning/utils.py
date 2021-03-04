"""
 Copyright (c) 2020 Intel Corporation
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
from typing import Tuple
from typing import Optional

import networkx as nx
import torch
import numpy as np

from nncf.layers import NNCF_DECONV_MODULES_DICT
from nncf.dynamic_graph.graph import PTNNCFNode
from nncf.dynamic_graph.context import Scope
from nncf.nncf_network import NNCFNetwork


def get_bn_node_for_conv(graph: nx.Graph, conv_node: dict) -> Optional[dict]:
    out_edges = graph.out_edges(conv_node['key'])
    for _, out_node_key in out_edges:
        out_node = graph.nodes[out_node_key]
        if out_node['op_exec_context'].operator_name == 'batch_norm':
            return out_node
    return None


def get_bn_for_module_scope(target_model: NNCFNetwork, module_scope: Scope) -> Tuple[torch.nn.Module, Scope]:
    """
    Returns batch norm module that corresponds to module_scope convolution.
    :param target_model: NNCFNetwork to work with
    :param module_scope:
    :return: batch norm module
    """
    graph = target_model.get_original_graph()
    module_graph_node = graph.find_node_in_nx_graph_by_scope(module_scope)
    # pylint:disable=protected-access
    bn_graph_node = get_bn_node_for_conv(graph._nx_graph, module_graph_node)
    bn_module = None
    if bn_graph_node:
        bn_scope = bn_graph_node['op_exec_context'].scope_in_model
        bn_module = target_model.get_module_by_scope(bn_graph_node['op_exec_context'].scope_in_model)
        return bn_module, bn_scope
    return bn_module, bn_graph_node


def pt_is_depthwise_conv(node: PTNNCFNode) -> bool:
    return node.module_attributes.groups == node.module_attributes.in_channels \
           and (node.module_attributes.out_channels % node.module_attributes.in_channels == 0) \
           and node.module_attributes.in_channels > 1


def pt_is_conv_with_downsampling(node: PTNNCFNode) -> bool:
    return not np.all(np.array(node.module_attributes.stride) == 1) \
           and node.node_type not in [deconv.op_func_name for deconv in NNCF_DECONV_MODULES_DICT]
