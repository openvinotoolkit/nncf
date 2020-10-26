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
import math
from collections import deque

import torch
from functools import partial

import networkx as nx
from nncf.layers import NNCF_CONV_MODULES_DICT

from nncf.dynamic_graph.context import Scope
from nncf.dynamic_graph.graph import NNCFGraph, NNCFNode
from nncf.nncf_network import NNCFNetwork


# pylint: disable=protected-access
def get_rounded_pruned_element_number(total, sparsity_rate, multiple_of=8):
    """
    Calculates number of sparsified elements (approximately sparsity rate) from total such as
    number of remaining items will be multiple of some value.
    Always rounds number of remaining elements up.
    :param total: total elements number
    :param sparsity_rate: prorortion of zero elements in total.
    :param multiple_of:
    :return: number of elements to be zeroed
    """
    remaining_elems = math.ceil((total - total * sparsity_rate) / multiple_of) * multiple_of
    return max(total - remaining_elems, 0)


def get_bn_node_for_conv(graph: nx.Graph, conv_node: dict):
    out_edges = graph.out_edges(conv_node['key'])
    for _, out_node_key in out_edges:
        out_node = graph.nodes[out_node_key]
        if out_node['op_exec_context'].operator_name == 'batch_norm':
            return out_node
    return None


def get_bn_for_module_scope(target_model: NNCFNetwork, module_scope: Scope):
    """
    Returns batch norm module that corresponds to module_scope convolution.
    :param target_model: NNCFNetwork to work with
    :param module_scope:
    :return: batch norm module
    """
    graph = target_model.get_original_graph()
    module_graph_node = graph.find_node_in_nx_graph_by_scope(module_scope)
    bn_graph_node = get_bn_node_for_conv(graph._nx_graph, module_graph_node)
    bn_module = None
    if bn_graph_node:
        bn_module = target_model.get_module_by_scope(bn_graph_node['op_exec_context'].scope_in_model)
    return bn_module


def find_first_ops_with_type(nncf_graph: NNCFGraph, nodes, required_types, forward: bool = True):
    """
    Looking for first nodes with type from pruned_ops_types that are reachable from nodes.
    :param nncf_graph: NNCFGraph to work with
    :param nodes: nodes from which search begins
    :param required_types: types of nodes for search
    :param forward: whether the search will be forward or backward
    :return:
    """
    graph = nncf_graph._nx_graph
    get_edges_fn = graph.out_edges if forward else graph.in_edges

    found_nodes = []
    visited = {n: False for n in graph.nodes}
    node_stack = deque(nodes)
    while node_stack:
        last_node = node_stack.pop()
        last_node_type = nncf_graph.node_type_fn(last_node)

        if not visited[last_node['key']]:
            visited[last_node['key']] = True
        else:
            continue

        if last_node_type not in required_types:
            edges = get_edges_fn(last_node['key'])
            for in_node_name, out_node_name in edges:
                cur_node = graph.nodes[out_node_name] if forward else graph.nodes[in_node_name]

                if not visited[cur_node['key']]:
                    node_stack.append(cur_node)
        else:
            found_nodes.append(last_node)
    return found_nodes


def traverse_function(node: NNCFNode, output, nncf_graph: NNCFGraph, required_types, visited):
    nx_node = nncf_graph._nx_graph.nodes[nncf_graph.get_node_key_by_id(node.node_id)]
    node_type = nncf_graph.node_type_fn(nx_node)
    if visited[node.node_id]:
        return True, output
    visited[node.node_id] = True

    if node_type not in required_types:
        return False, output

    output.append(node)
    return True, output


def get_first_pruned_modules(target_model: NNCFNetwork, pruned_ops_types):
    """
    Looking for first pruned modules in target model.
    First == layer of pruned type, that there is a path from the input such that there are no other
    pruned operations on it.
    :param pruned_ops_types: types of modules that will be pruned
    :param target_model: model to work with
    :return: list of all first pruned modules
    """
    graph = target_model.get_original_graph()  # NNCFGraph here
    graph_roots = graph.get_input_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_idxs()}
    partial_traverse_function = partial(traverse_function, nncf_graph=graph, required_types=pruned_ops_types,
                                        visited=visited)

    first_pruned_nodes = []
    for root in graph_roots:
        first_pruned_nodes.extend(graph.traverse_graph(root, partial_traverse_function))
    first_pruned_modules = [target_model.get_module_by_scope(n.op_exec_context.scope_in_model)
                            for n in first_pruned_nodes]
    return first_pruned_modules


def get_last_pruned_modules(target_model: NNCFNetwork, pruned_ops_types):
    """
    Looking for last pruned modules in target model.
    Last == layer of pruned type, that there is a path from this layer to the model output
    such that there are no other pruned operations on it.
    :param pruned_ops_types: types of modules that will be pruned
    :param target_model: model to work with
    :return: list of all last pruned modules
    """
    graph = target_model.get_original_graph()  # NNCFGraph here
    graph_outputs = graph.get_graph_outputs()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_idxs()}
    partial_traverse_function = partial(traverse_function, nncf_graph=graph, required_types=pruned_ops_types,
                                        visited=visited)
    last_pruned_nodes = []
    for output in graph_outputs:
        last_pruned_nodes.extend(graph.traverse_graph(output, partial_traverse_function, False))

    last_pruned_modules = [target_model.get_module_by_scope(n.op_exec_context.scope_in_model)
                           for n in last_pruned_nodes]
    return last_pruned_modules


def get_sources_of_node(nncf_node: NNCFNode, graph: NNCFGraph, sources_types):
    """
    Source is a node of sourse such that there is path from this node to nx_node and on this path
    no node has one of sources_types type.
    :param sources_types: list of sources types
    :param nncf_node: NNCFNode to get sources
    :param graph: NNCF graph to work with
    :return: list of all sources nodes
    """
    visited = {node_id: False for node_id in graph.get_all_node_idxs()}
    partial_traverse_function = partial(traverse_function, nncf_graph=graph, required_types=sources_types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.op_exec_context.operator_name in sources_types:
        nncf_nodes = graph.get_previous_nodes(nncf_node)

    source_nodes = []
    for node in nncf_nodes:
        source_nodes.extend(graph.traverse_graph(node, partial_traverse_function, False))
    return source_nodes


def is_conv_with_downsampling(conv_module):
    return not torch.all(torch.tensor(conv_module.stride) == 1)


def is_grouped_conv(conv_module):
    return conv_module.groups != 1


def is_depthwise_conv(conv_module):
    return conv_module.groups == conv_module.in_channels and (conv_module.out_channels % conv_module.in_channels == 0)


def get_previous_conv(target_model: NNCFNetwork, module, module_scope):
    """
    Return source convolution of module. If node has other source type or there are more than one source - return None.
    """
    conv_types = [str.lower(v.__name__) for v in NNCF_CONV_MODULES_DICT.values()]

    graph = target_model.get_original_graph()
    nx_node = graph.find_node_in_nx_graph_by_scope(module_scope)
    nncf_node = graph._nx_node_to_nncf_node(nx_node)
    sources = get_sources_of_node(nncf_node, graph, conv_types + ['linear'])
    if len(sources) == 1 and sources[0].op_exec_context.operator_name in conv_types:
        return sources[0]
    return None


def _find_next_nodes_of_types(model, nncf_node, types):
    sources_types = types
    graph = model.get_original_graph()
    visited = {node_id: False for node_id in graph.get_all_node_idxs()}
    partial_traverse_function = partial(traverse_function, nncf_graph=graph, required_types=sources_types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.op_exec_context.operator_name in sources_types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes