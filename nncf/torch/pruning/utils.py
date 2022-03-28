"""
 Copyright (c) 2022 Intel Corporation
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
from typing import Dict
from typing import Optional
from typing import List
from typing import Tuple

import torch

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.torch.graph.graph import NNCFNode
from nncf.torch.layers import NNCF_GENERAL_CONV_MODULES_DICT
from nncf.torch.layers import NNCF_LINEAR_MODULES_DICT
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.nncf_network import NNCFNetwork
from nncf.common.utils.logger import logger as nncf_logger


def get_bn_node_for_conv(graph: NNCFGraph, conv_node: NNCFNode) -> Optional[NNCFNode]:
    successors = graph.get_next_nodes(conv_node)
    for succ in successors:
        if succ.node_type == 'batch_norm':
            return succ
    return None


def get_bn_for_conv_node_by_name(target_model: NNCFNetwork, conv_node_name: NNCFNodeName) -> Optional[torch.nn.Module]:
    """
    Returns a batch norm module in target_model that corresponds immediately following a given
    convolution node in the model's NNCFGraph representation.
    :param target_model: NNCFNetwork to work with
    :param module_scope:
    :return: batch norm module
    """
    graph = target_model.get_original_graph()
    conv_node = graph.get_node_by_name(conv_node_name)
    bn_node = get_bn_node_for_conv(graph, conv_node)
    if bn_node is None:
        return None
    bn_module = target_model.get_containing_module(bn_node.node_name)
    return bn_module


def init_output_masks_in_graph(graph: NNCFGraph, nodes: List):
    """
    Initialize masks in graph for mask propagation algorithm

    :param graph: NNCFNetwork
    :param nodes: list with pruned nodes
    """
    for node in graph.get_all_nodes():
        node.data.pop('output_mask', None)

    for minfo in nodes:
        mask = minfo.operand.binary_filter_pruning_mask
        nncf_node = graph.get_node_by_id(minfo.nncf_node_id)
        nncf_node.data['output_mask'] = PTNNCFTensor(mask)


def _calculate_output_shape(graph: NNCFGraph, node: NNCFNode) -> Tuple[int, ...]:
    """
    Calculates output shape of convolution layer by input edge.

    :param graph: the model graph
    :param node: node from NNCF graph
    :return: output shape
    """
    in_edge = graph.get_input_edges(node)[0]
    shape = list(in_edge.tensor_shape)[2:]
    attrs = node.layer_attributes

    assert isinstance(attrs, ConvolutionLayerAttributes)

    for i, _ in enumerate(shape):
        if attrs.transpose:
            shape[i] = (shape[i] - 1) * attrs.stride[i] - 2 * attrs.padding_values[i] + attrs.kernel_size[i]
        else:
            shape[i] = (shape[i] + 2 * attrs.padding_values[i] - attrs.kernel_size[i]) // attrs.stride[i] + 1
    return tuple(shape)


def collect_output_shapes(graph: NNCFGraph) -> Dict[NNCFNodeName, List[int]]:
    """
    Collects output dimension shapes for convolutions and fully connected layers
    from the connected edges in the NNCFGraph.

    :param graph: NNCFGraph.
    :return: Dictionary of output dimension shapes. E.g {node_name: (height, width)}
    """
    modules_out_shapes = {}
    for node in graph.get_nodes_by_types([v.op_func_name for v in NNCF_GENERAL_CONV_MODULES_DICT]):
        output_edges = graph.get_output_edges(node)
        if output_edges:
            out_edge = output_edges[0]
            out_shape = out_edge.tensor_shape[2:]
        else:
            # For disconnected NNCFGraph when node have no output edge
            out_shape = _calculate_output_shape(graph, node)
            nncf_logger.error("Node %s have no output edge in NNCFGraph", node.node_name)
        modules_out_shapes[node.node_name] = out_shape

    for node in graph.get_nodes_by_types([v.op_func_name for v in NNCF_LINEAR_MODULES_DICT]):
        output_edges = graph.get_output_edges(node)
        if output_edges:
            out_edge = graph.get_output_edges(node)[0]
            out_shape = out_edge.tensor_shape
            modules_out_shapes[node.node_name] = out_shape[-1]
        else:
            # For disconnected NNCFGraph when node have no output edge
            nncf_logger.error("Node %s have no output edge in NNCFGraph", node.node_name)
            modules_out_shapes[node.node_name] = node.layer_attributes.out_features
    return modules_out_shapes


def collect_input_shapes(graph: NNCFGraph) -> Dict[NNCFNodeName, List[int]]:
    """
    Collects input dimension shapes for fully connected layers from the connected edges in the NNCFGraph.

    :param graph: NNCFGraph.
    :return: Dictionary of input dimension shapes. E.g {node_name: (height, width)}
    """
    modules_in_shapes = {}
    for node in graph.get_nodes_by_types([v.op_func_name for v in NNCF_LINEAR_MODULES_DICT]):
        in_edge = graph.get_input_edges(node)[0]
        in_shape = in_edge.tensor_shape
        if len(in_shape) == 1:
            modules_in_shapes[node.node_name] = in_shape[0]
        else:
            modules_in_shapes[node.node_name] = in_shape[1:]
    return modules_in_shapes
