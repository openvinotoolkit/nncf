"""
 Copyright (c) 2021 Intel Corporation
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

from functools import partial
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.utils.registry import Registry


def is_grouped_conv(node: NNCFNode) -> bool:
    return isinstance(node.layer_attributes, ConvolutionLayerAttributes) \
           and node.layer_attributes.groups != 1


def get_sources_of_node(nncf_node: NNCFNode, graph: NNCFGraph, sources_types: List[str]) -> List[NNCFNode]:
    """
    Source is a node of source such that there is path from this node to `nncf_node` and on this path
    no node has one of `sources_types` type.

    :param sources_types: List of sources types.
    :param nncf_node: NNCFNode to get sources.
    :param graph: NNCF graph to work with.
    :return: List of all sources nodes.
    """
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in sources_types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type in sources_types:
        nncf_nodes = graph.get_previous_nodes(nncf_node)

    source_nodes = []
    for node in nncf_nodes:
        source_nodes.extend(graph.traverse_graph(node, partial_traverse_function, False))
    return source_nodes


def find_next_nodes_not_of_types(graph: NNCFGraph, nncf_node: NNCFNode, types: List[str]) -> List[NNCFNode]:
    """
    Traverse nodes in the graph from nncf node to find first nodes that aren't of type from types list.
    First nodes with some condition mean nodes:
        - for which this condition is true
        - reachable from `nncf_node` such that on the path from `nncf_node` to
          this nodes there are no other nodes with fulfilled condition

    :param graph: Graph to work with.
    :param nncf_node: NNCFNode to start search.
    :param types: List of types.
    :return: List of next nodes for nncf_node of type not from types list.
    """
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x not in types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type not in types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes


def get_next_nodes_of_types(graph: NNCFGraph, nncf_node: NNCFNode, types: List[str]) -> List[NNCFNode]:
    """
    Looking for nodes with type from types list from `nncf_node` such that there is path
    from `nncf_node` to this node and on this path no node has one of types type.

    :param graph: Graph to work with.
    :param nncf_node: NNCFNode to start search.
    :param types: List of types to find.
    :return: List of next nodes of nncf_node with type from types list.
    """
    sources_types = types
    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function, type_check_fn=lambda x: x in sources_types,
                                        visited=visited)
    nncf_nodes = [nncf_node]
    if nncf_node.node_type in sources_types:
        nncf_nodes = graph.get_next_nodes(nncf_node)

    next_nodes = []
    for node in nncf_nodes:
        next_nodes.extend(graph.traverse_graph(node, partial_traverse_function))
    return next_nodes


def get_rounded_pruned_element_number(total: int, sparsity_rate: float, multiple_of: int = 8) -> int:
    """
    Calculates number of sparsified elements (approximately sparsity rate) from total such as
    number of remaining items will be multiple of some value.
    Always rounds number of remaining elements up.

    :param total: Total elements number.
    :param sparsity_rate: Prorortion of zero elements in total.
    :param multiple_of: Number of remaining elements must be a multiple of `multiple_of`.
    :return: Number of elements to be zeroed.
    """
    remaining_elems = math.ceil((total - total * sparsity_rate) / multiple_of) * multiple_of
    return max(total - remaining_elems, 0)


def traverse_function(node: NNCFNode, output: List[NNCFNode], type_check_fn, visited) \
        -> Tuple[bool, List[NNCFNode]]:
    if visited[node.node_id]:
        return True, output
    visited[node.node_id] = True

    if not type_check_fn(node.node_type):
        return False, output

    output.append(node)
    return True, output


def get_first_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for first node in graph with type in `op_types`.
    First == layer with type in `op_types`, that there is a path from the input such that there are no other
    operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all first nodes with type in `op_types`.
    """
    graph_roots = graph.get_input_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function,
                                        type_check_fn=lambda x: x in op_types,
                                        visited=visited)

    first_nodes_of_type = []
    for root in graph_roots:
        first_nodes_of_type.extend(graph.traverse_graph(root, partial_traverse_function))
    return first_nodes_of_type


def get_last_nodes_of_type(graph: NNCFGraph, op_types: List[str]) -> List[NNCFNode]:
    """
    Looking for last node in graph with type in `op_types`.
    Last == layer with type in `op_types`, that there is a path from this layer to the model output
    such that there are no other operations with type in `op_types` on it.

    :param op_types: Types of modules to track.
    :param graph: Graph to work with.
    :return: List of all last pruned nodes.
    """
    graph_outputs = graph.get_output_nodes()  # NNCFNodes here

    visited = {node_id: False for node_id in graph.get_all_node_ids()}
    partial_traverse_function = partial(traverse_function,
                                        type_check_fn=lambda x: x in op_types,
                                        visited=visited)
    last_nodes_of_type = []
    for output in graph_outputs:
        last_nodes_of_type.extend(graph.traverse_graph(output, partial_traverse_function, False))

    return last_nodes_of_type


def get_previous_conv(graph: NNCFGraph, nncf_node: NNCFNode,
                      pruning_types: List[str], stop_propagation_ops: List[str]) -> Optional[NNCFNode]:
    """
    Returns source convolution of the node. If the node has another source type or there is
    more than one source - returns None.

    :return: Source convolution of node. If the node has another source type or there is more
        than one source - returns None.
    """
    sources = get_sources_of_node(nncf_node, graph, pruning_types + stop_propagation_ops)
    if len(sources) == 1 and sources[0].node_type in pruning_types:
        return sources[0]
    return None


def get_conv_in_out_channels(graph: NNCFGraph):
    """
    Collects the number of input and output channels for each convolution in the graph.

    :param graph: NNCFGraph
    :return Dictionary with the number of input channels to convolution layers:
            {node_name: input_channels_num}
            Dictionary with the number of output channels from convolution layers:
            {node_name: output_channels_num}
    """
    in_channels, out_channels = {}, {}
    for node in graph.get_all_nodes():
        if isinstance(node.layer_attributes, ConvolutionLayerAttributes):
            name = node.node_name
            if name in in_channels and name in out_channels:
                continue
            in_channels[name] = node.layer_attributes.in_channels
            out_channels[name] = node.layer_attributes.out_channels
    return in_channels, out_channels


def get_cluster_next_nodes(graph: NNCFGraph, pruned_groups_info: Clusterization[PrunedLayerInfoBase],
                           prunable_types: List[str]) -> Dict[int, List[NNCFNodeName]]:
    """
    Finds nodes of `prunable_types` types that receive the output of a pruned cluster as input.

    :param graph: NNCFGraph.
    :param pruned_groups_info: `Clusterization` of pruning groups.
    :param prunable_types: Types of nodes that will be returned.
    :return Dictionary of next node names by cluster {cluster_id: [node_name]}.
    """
    next_nodes = {}
    for cluster in pruned_groups_info.get_all_clusters():
        next_nodes_cluster = set()
        cluster_nodes = set()
        for pruned_layer_info in cluster.elements:
            nncf_cluster_node = graph.get_node_by_id(pruned_layer_info.nncf_node_id)
            cluster_nodes.add(nncf_cluster_node.node_name)
            curr_next_nodes = get_next_nodes_of_types(graph, nncf_cluster_node, prunable_types)

            next_nodes_idxs = [n.node_name for n in curr_next_nodes]
            next_nodes_cluster = next_nodes_cluster.union(next_nodes_idxs)
        next_nodes[cluster.id] = list(next_nodes_cluster - cluster_nodes)
    return next_nodes


def count_flops_and_weights(graph: NNCFGraph,
                            input_shapes: Dict[NNCFNodeName, List[int]],
                            output_shapes: Dict[NNCFNodeName, List[int]],
                            conv_op_metatypes: List[Type[OperatorMetatype]],
                            linear_op_metatypes: List[Type[OperatorMetatype]],
                            input_channels: Dict[NNCFNodeName, int] = None,
                            output_channels: Dict[NNCFNodeName, int] = None) -> Tuple[int, int]:
    """
    Counts the number weights and FLOPs in the model for convolution and fully connected layers.

    :param graph: NNCFGraph.
    :param input_shapes: Dictionary of input dimension shapes for convolutions and
        fully connected layers. E.g {node_name: (height, width)}
    :param output_shapes: Dictionary of output dimension shapes for convolutions and
        fully connected layers. E.g {node_name: (height, width)}
    :param conv_op_metatypes: List of metatypes defining convolution operations.
    :param linear_op_metatypes: List of metatypes defining linear/fully connected operations.
    :param input_channels: Dictionary of input channels number in convolutions.
        If not specified, taken from the graph. {node_name: channels_num}
    :param output_channels: Dictionary of output channels number in convolutions.
        If not specified, taken from the graph. {node_name: channels_num}
    :return number of FLOPs for the model
            number of weights (params) in the model
    """
    flops_pers_node, weights_per_node = count_flops_and_weights_per_node(graph,
                                                                         input_shapes, output_shapes,
                                                                         conv_op_metatypes, linear_op_metatypes,
                                                                         input_channels, output_channels)
    return sum(flops_pers_node.values()), sum(weights_per_node.values())


def count_flops_and_weights_per_node(graph: NNCFGraph,
                                     input_shapes: Dict[NNCFNodeName, List[int]],
                                     output_shapes: Dict[NNCFNodeName, List[int]],
                                     conv_op_metatypes: List[Type[OperatorMetatype]],
                                     linear_op_metatypes: List[Type[OperatorMetatype]],
                                     input_channels: Dict[NNCFNodeName, int] = None,
                                     output_channels: Dict[NNCFNodeName, int] = None) -> \
        Tuple[Dict[NNCFNodeName, int], Dict[NNCFNodeName, int]]:
    """
    Counts the number weights and FLOPs per node in the model for convolution and fully connected layers.

    :param graph: NNCFGraph.
    :param input_shapes: Dictionary of input dimension shapes for convolutions and
        fully connected layers. E.g {node_name: (height, width)}
    :param output_shapes: Dictionary of output dimension shapes for convolutions and
        fully connected layers. E.g {node_name: (height, width)}
    :param conv_op_metatypes: List of metatypes defining convolution operations.
    :param linear_op_metatypes: List of metatypes defining linear/fully connected operations.
    :param input_channels: Dictionary of input channels number in convolutions.
        If not specified, taken from the graph. {node_name: channels_num}
    :param output_channels: Dictionary of output channels number in convolutions.
        If not specified, taken from the graph. {node_name: channels_num}
    :return Dictionary of FLOPs number {node_name: flops_num}
            Dictionary of weights number {node_name: weights_num}
    """
    flops = {}
    weights = {}
    input_channels = input_channels or {}
    output_channels = output_channels or {}
    for node in graph.get_nodes_by_metatypes(conv_op_metatypes):
        name = node.node_name
        num_in_channels = input_channels.get(name, node.layer_attributes.in_channels)
        num_out_channels = output_channels.get(name, node.layer_attributes.out_channels)
        flops_numpy = 2 * np.prod(node.layer_attributes.kernel_size) * \
                      num_in_channels * num_out_channels * np.prod(output_shapes[name])
        weights_numpy = np.prod(node.layer_attributes.kernel_size) * num_in_channels * num_out_channels
        flops[name] = flops_numpy.astype(int).item()
        weights[name] = weights_numpy.astype(int).item()

    for node in graph.get_nodes_by_metatypes(linear_op_metatypes):
        name = node.node_name
        flops_numpy = 2 * np.prod(input_shapes[name]) * np.prod(output_shapes[name])
        weights_numpy = np.prod(input_shapes[name]) * np.prod(output_shapes[name])
        flops[name] = flops_numpy.astype(int).item()
        weights[name] = weights_numpy.astype(int).item()

    return flops, weights


def calculate_in_out_channels_in_uniformly_pruned_model(pruning_groups: List[Cluster[PrunedLayerInfoBase]],
                                                        pruning_rate: float,
                                                        full_input_channels: Dict[str, int],
                                                        full_output_channels: Dict[str, int],
                                                        pruning_groups_next_nodes: Dict[int, List[str]]):
    """
    Imitates filters pruning by removing `pruning_rate` percent of output filters in each pruning group
    and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

    :param pruning_groups: A list of pruning groups.
    :param pruning_rate: Target pruning rate.
    :param full_input_channels:  A dictionary of input channels number in original model.
    :param full_output_channels: A dictionary of output channels number in original model.
    :param pruning_groups_next_nodes: A dictionary of next nodes of each pruning group.
    :return Dictionary of new input channels number {node_name: channels_num}
    :return Dictionary of new output channels number {node_name: channels_num}
    """
    tmp_in_channels = full_input_channels.copy()
    tmp_out_channels = full_output_channels.copy()

    for group in pruning_groups:
        layer_name = group.elements[0].node_name
        assert all(tmp_out_channels[layer_name] == tmp_out_channels[node.node_name] for node in
                   group.elements)
        # Prune all nodes in cluster (by output channels)
        old_out_channels = full_output_channels[layer_name]
        num_of_sparse_elems = get_rounded_pruned_element_number(old_out_channels, pruning_rate)
        new_out_channels_num = old_out_channels - num_of_sparse_elems

        for minfo in group.elements:
            tmp_out_channels[minfo.node_name] = new_out_channels_num

        # Prune in_channels in all next nodes of cluster
        for node_name in pruning_groups_next_nodes[group.id]:
            tmp_in_channels[node_name] -= num_of_sparse_elems

    return tmp_in_channels, tmp_out_channels


class PruningOperationsMetatypeRegistry(Registry):
    def __init__(self, name):
        super().__init__(name)
        self._op_name_to_op_class = {}

    def register(self, name=None):
        name_ = name
        super_register = super()._register

        def wrap(obj):
            cls_name = name_
            if cls_name is None:
                cls_name = obj.__name__

            super_register(obj, cls_name)
            op_names = obj.get_all_op_aliases()
            for name in op_names:
                if name not in self._op_name_to_op_class:
                    self._op_name_to_op_class[name] = obj
                else:
                    assert self._op_name_to_op_class[name] == obj, \
                        'Inconsistent operator type registry - single patched op name maps to multiple metatypes!'
            return obj

        return wrap

    def get_operator_metatype_by_op_name(self, op_name: str):
        if op_name in self._op_name_to_op_class:
            return self._op_name_to_op_class[op_name]
        return None
