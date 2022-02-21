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

from functools import partial
from typing import Dict, List, Optional, Tuple, Type, Union, Callable
from enum import Enum

import math
import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.layer_attributes import LinearLayerAttributes
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.tensor import NNCFTensor
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


def get_previous_convs(graph: NNCFGraph, nncf_node: NNCFNode,
                       pruning_types: List[str], stop_propagation_ops: List[str]) -> List[NNCFNode]:
    """
    Returns source convolutions of the node.

    :return: List of source convolutions of node.
    """
    sources = get_sources_of_node(nncf_node, graph, pruning_types + stop_propagation_ops)
    sources = [source for source in sources if source.node_type in pruning_types]
    return sources


def get_prunable_layers_in_out_channels(graph: NNCFGraph):
    """
    Collects the number of input and output channels for each prunable layer in the graph.

    :param graph: NNCFGraph
    :return Dictionary with the number of input channels to convolution layers:
            {node_name: input_channels_num}
            Dictionary with the number of output channels from convolution layers:
            {node_name: output_channels_num}
    """
    in_channels, out_channels = {}, {}
    for node in graph.get_all_nodes():
        if isinstance(node.layer_attributes, (ConvolutionLayerAttributes, LinearLayerAttributes)):
            name = node.node_name
            if name in in_channels and name in out_channels:
                continue
            in_channels[name] = get_input_channels(node)
            out_channels[name] = get_output_channels(node)
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
        if is_prunable_depthwise_conv(node):
            # Prunable depthwise conv processed in special way
            # because common way to calculate filters per
            # channel for such layer leads to zero in case
            # some of the output channels are pruned.
            filters_per_channel = 1
        else:
            filters_per_channel = num_out_channels // node.layer_attributes.groups

        flops_numpy = 2 * np.prod(node.layer_attributes.kernel_size) * \
                      num_in_channels * filters_per_channel * np.prod(output_shapes[name])
        weights_numpy = np.prod(node.layer_attributes.kernel_size) * num_in_channels * filters_per_channel
        flops[name] = flops_numpy.astype(int).item()
        weights[name] = weights_numpy.astype(int).item()

    for node in graph.get_nodes_by_metatypes(linear_op_metatypes):
        name = node.node_name
        flops_numpy = 2 * np.prod(input_shapes[name]) * np.prod(output_shapes[name])
        weights_numpy = np.prod(input_shapes[name]) * np.prod(output_shapes[name])
        flops[name] = flops_numpy.astype(int).item()
        weights[name] = weights_numpy.astype(int).item()

    return flops, weights


def count_filters_num(graph: NNCFGraph,
                      op_metatypes: List[Type[OperatorMetatype]],
                      output_channels: Dict[NNCFNodeName, int] = None) -> int:
    """
    Counts filters of `op_metatypes` layers taking into account new output channels number.

    :param graph: Graph to work with.
    :param op_metatypes: List of metatypes defining convolution operations.
    :param output_channels:  A dictionary of output channels number in pruned model.
    :return: Current number of filters according to given graph and output channels.
    """
    filters_num = 0
    output_channels = output_channels or {}
    for node in graph.get_nodes_by_metatypes(op_metatypes):
        filters_num += output_channels.get(node.node_name, get_output_channels(node))
    return filters_num


def _calculate_in_out_channels(pruning_groups: List[Cluster[PrunedLayerInfoBase]],
                               sparse_elements_counter: Callable[[str], int],
                               full_input_channels: Dict[str, int],
                               full_output_channels: Dict[str, int],
                               pruning_groups_next_nodes: Dict[int, List[str]]) -> Tuple[Dict[str, int],
                                                                                         Dict[str, int]]:
    tmp_in_channels = full_input_channels.copy()
    tmp_out_channels = full_output_channels.copy()

    for group in pruning_groups:
        layer_name = group.elements[0].node_name
        assert all(tmp_out_channels[layer_name] == tmp_out_channels[node.node_name] for node in
                   group.elements)
        # Prune all nodes in cluster (by output channels)
        old_out_channels = full_output_channels[layer_name]
        num_of_sparse_elems = sparse_elements_counter(layer_name)
        new_out_channels_num = old_out_channels - num_of_sparse_elems

        for minfo in group.elements:
            tmp_out_channels[minfo.node_name] = new_out_channels_num
            if minfo.is_depthwise:
                tmp_in_channels[minfo.node_name] = new_out_channels_num

        # Prune in_channels in all next nodes of cluster
        for node_name in pruning_groups_next_nodes[group.id]:
            tmp_in_channels[node_name] -= num_of_sparse_elems

    return tmp_in_channels, tmp_out_channels


def calculate_in_out_channels_in_uniformly_pruned_model(pruning_groups: List[Cluster[PrunedLayerInfoBase]],
                                                        pruning_level: float,
                                                        full_input_channels: Dict[str, int],
                                                        full_output_channels: Dict[str, int],
                                                        pruning_groups_next_nodes: Dict[int, List[str]]) -> \
                                                        Tuple[Dict[str, int], Dict[str, int]]:
    """
    Imitates filters pruning by removing `pruning_rate` percent of output filters in each pruning group
    and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

    :param pruning_groups: A list of pruning groups.
    :param pruning_level: Target pruning rate.
    :param full_input_channels:  A dictionary of input channels number in original model.
    :param full_output_channels: A dictionary of output channels number in original model.
    :param pruning_groups_next_nodes: A dictionary of next nodes of each pruning group.
    :return Dictionary of new input channels number {node_name: channels_num}
    :return Dictionary of new output channels number {node_name: channels_num}
    """
    def get_num_of_sparse_elements_by_node(node_name: str) -> int:
        old_out_channels = full_output_channels[node_name]
        return get_rounded_pruned_element_number(old_out_channels, pruning_level)

    return _calculate_in_out_channels(pruning_groups,
                                      get_num_of_sparse_elements_by_node,
                                      full_input_channels,
                                      full_output_channels,
                                      pruning_groups_next_nodes)


def calculate_in_out_channels_by_masks(pruning_groups: List[Cluster[PrunedLayerInfoBase]],
                                       num_of_sparse_elements_by_node: Dict[NNCFNodeName, int],
                                       full_input_channels: Dict[str, int],
                                       full_output_channels: Dict[str, int],
                                       pruning_groups_next_nodes: Dict[int, List[str]]) -> Tuple[Dict[str, int],
                                                                                                 Dict[str, int]]:
    """
    Imitates filters pruning by removing output filters zeroed by pruning masks in each pruning group
    and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

    :param pruning_groups: A list of pruning groups.
    :param num_of_sparse_elements_by_node: A dictionary of num_of_sparse_elements of each pruning node.
    :param full_input_channels:  A dictionary of input channels number in original model.
    :param full_output_channels: A dictionary of output channels number in original model.
    :param pruning_groups_next_nodes: A dictionary of next nodes of each pruning group.
    :return Dictionary of new input channels number {node_name: channels_num}
    """
    def get_num_of_sparse_elements_by_node(node_name: str) -> int:
        return num_of_sparse_elements_by_node[node_name]

    return _calculate_in_out_channels(pruning_groups,
                                      get_num_of_sparse_elements_by_node,
                                      full_input_channels,
                                      full_output_channels,
                                      pruning_groups_next_nodes)


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


class PruningAnalysisReason(Enum):
    """
    Enum of possible pruning analysis decisions reasons.
    """

    IGNORED_SCOPE = 'node in ignored scope'
    FIRST_CONV = 'this scope is one of the first convolutions'
    LAST_CONV = 'this scope is convolution with output which directly affects model output dimensions'
    GROUP_CONV = 'this scope is grouped convolution'
    DOWNSAMPLE_CONV = 'this scope is convolution with downsample'
    MODEL_ANALYSIS = 'of model analysis'
    DIMENSION_MISMATCH = 'of dimension mismatch'
    CLOSING_CONV_MISSING = 'closing convolution missing'
    IN_GROUP_OF_UNPRUNABLE = 'is in the group with non prunable layers'

    @classmethod
    def message(cls, node_name: str, decision: Optional['PruningAnalysisDecision']) -> str:
        """
        Returns the node pruning analysis decisions in a human-readable format.

        :param node_name: Name of given node.
        :param decision: Pruning analysis decision for given node.
        :return: Pruning analysis decision in a human-readable format.
        """
        prefix = f'ignored adding Weight Pruner in: {node_name}'
        reasons = decision.reasons
        if not reasons:
            return prefix
        # Filter messages
        if len(reasons) > 1 and cls.CLOSING_CONV_MISSING in reasons:
            reasons.remove(cls.CLOSING_CONV_MISSING)
        if len(reasons) == 1 and cls.IN_GROUP_OF_UNPRUNABLE in reasons:
            return ''
        return prefix + ' because ' + ' and '.join([reason.value for reason in reasons])


class PruningAnalysisDecision:
    """
    Container for pruning analysis decisions. Contains decision which is boolean marker either
    node prunable or not (prunable if decision attribute is True) and
    pruning analysis reason in PruningAnalysisReason format. In case of positive
    decision (decision == True) possible reason will be ignored.
    """

    def __init__(self,
                 decision: bool,
                 possible_reasons: Optional[Union[List[PruningAnalysisReason], PruningAnalysisReason]] = None):
        self.decision = decision
        if not isinstance(possible_reasons, list):
            possible_reasons = [possible_reasons]
        self._reasons = possible_reasons if not decision and possible_reasons else None \
            # type: Optional[List[PruningAnalysisReason]]

    def __repr__(self) -> str:
        representation = f'Prunable: {self.decision}'
        if not self.decision:
            representation += '; Reasons: ' + str(self._reasons)
        return representation

    def __eq__(self, other: 'PruningAnalysisDecision') -> bool:
        eq = self.decision == other.decision
        if self._reasons is None:
            return eq and other._reasons is None
        if other._reasons is None:
            return False
        return eq and set(self._reasons) == set(other._reasons)

    def __bool__(self) -> bool:
        return self.decision

    @property
    def reasons(self) -> Optional[List[PruningAnalysisReason]]:
        if self._reasons:
            return self._reasons.copy()
        return None

    def join(self, other: 'PruningAnalysisDecision') -> 'PruningAnalysisDecision':
        """
        Join two pruning analysis decisions about one NNCFNode.

        :param other: pruning analysis decision to join with.
        :return: Joint pruning analysis decision.
        """
        if self.decision and other.decision:
            return self

        reasons = []
        for decision in [self, other]:
            if decision.reasons:
                reasons.extend(decision.reasons)

        return PruningAnalysisDecision(False, reasons)


def is_prunable_depthwise_conv(node: NNCFNode) -> bool:
    # Only convolutions with in_channels == groups == out_channels are supported
    # by pruning algorithm. Depthwise convolutions support ticket: #68580
    return isinstance(node.layer_attributes, ConvolutionLayerAttributes) \
           and node.layer_attributes.groups == node.layer_attributes.in_channels \
           and (node.layer_attributes.out_channels == node.layer_attributes.in_channels) \
           and node.layer_attributes.in_channels > 1


def is_conv_with_downsampling(node: NNCFNode) -> bool:
    layer_attrs = node.layer_attributes
    if isinstance(layer_attrs, ConvolutionLayerAttributes):
        return not np.all(np.array(layer_attrs.stride) == 1) \
           and not layer_attrs.transpose
    return False


def get_input_masks(node: NNCFNode, graph: NNCFGraph) -> List[Optional[NNCFTensor]]:
    """
    Returns input masks for all inputs of given NNCFNode.

    :param node: Given NNCFNode.
    :param graph: Graph to work with.
    :return: Input masks.
    """
    input_masks = [input_node.data['output_mask'] for input_node in graph.get_previous_nodes(node)]
    return input_masks


def get_input_channels(node: NNCFNode) -> int:
    """
    Returns count of input channels of an prunable node.

    :param node: Given prunable node.
    :return: Count of input channels of the given node.
    """
    layer_attrs = node.layer_attributes # type: Union[ConvolutionLayerAttributes, LinearLayerAttributes]
    if isinstance(layer_attrs, ConvolutionLayerAttributes):
        return layer_attrs.in_channels
    if isinstance(layer_attrs, LinearLayerAttributes):
        return layer_attrs.in_features
    raise RuntimeError(f'Can\'t get count of input channels from node {node}')


def get_output_channels(node: NNCFNode) -> int:
    """
    Returns count of output channels of an prunable node.

    :param node: Given prunable node.
    :return: Count of output channels of the given node.
    """
    layer_attrs = node.layer_attributes # type: Union[ConvolutionLayerAttributes, LinearLayerAttributes]
    if isinstance(layer_attrs, ConvolutionLayerAttributes):
        return layer_attrs.out_channels
    if isinstance(layer_attrs, LinearLayerAttributes):
        return layer_attrs.out_features
    raise RuntimeError(f'Can\'t get count of output channels from node {node}')


def identity_mask_propagation(node: NNCFNode, graph: NNCFGraph) -> None:
    """
    Propagates input mask through NNCFNode.

    :param node: Graph node to perform identity mask propagation on.
    :param graph: Graph to work with.
    """
    input_masks = get_input_masks(node, graph)
    if not input_masks:
        # In case for disconnected NNCFGraph
        input_masks = [None]
    assert len(input_masks) == 1
    node.data['output_mask'] = input_masks[0]
