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
from typing import List

import torch
import numpy as np

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.layers import NNCF_DECONV_MODULES_DICT
from nncf.graph.graph import PTNNCFNode
from nncf.dynamic_graph.context import Scope
from nncf.nncf_network import NNCFNetwork
from nncf.common.graph.module_attributes import ConvolutionModuleAttributes


def get_bn_node_for_conv(graph: NNCFGraph, conv_node: NNCFNode) -> Optional[NNCFNode]:
    successors = graph.get_successor_nncf_nodes(conv_node.node_id)
    for succ in successors:
        if succ.ia_op_exec_context.operator_name == 'batch_norm':
            return succ
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
    bn_graph_node = get_bn_node_for_conv(graph, module_graph_node)
    if bn_graph_node:
        bn_scope = bn_graph_node.ia_op_exec_context.scope_in_model
        bn_module = target_model.get_module_by_scope(bn_scope)
        return bn_module, bn_scope
    return None, None


def is_depthwise_conv(node: PTNNCFNode) -> bool:
    return isinstance(node.module_attributes, ConvolutionModuleAttributes) \
           and node.module_attributes.groups == node.module_attributes.in_channels \
           and (node.module_attributes.out_channels % node.module_attributes.in_channels == 0) \
           and node.module_attributes.in_channels > 1


def is_conv_with_downsampling(node: PTNNCFNode) -> bool:
    return isinstance(node.module_attributes, ConvolutionModuleAttributes) \
           and not np.all(np.array(node.module_attributes.stride) == 1) \
           and node.node_type not in [deconv.op_func_name for deconv in NNCF_DECONV_MODULES_DICT]


def init_output_masks_in_graph(graph: NNCFGraph, nodes: List):
    """
    Initialize masks in groph for mask propagation algorithm

    :param graph: NNCFNetwork
    :param nodes: list with pruned nodes
    """
    for node in graph.get_all_nodes():
        node.data.pop('input_mask', None)
        node.data.pop('output_mask', None)

    for minfo in nodes:
        mask = minfo.operand.binary_filter_pruning_mask
        nncf_node = graph.get_nncf_node_by_id(minfo.nncf_node_id)
        nncf_node.data['output_mask'] = mask
