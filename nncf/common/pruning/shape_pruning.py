
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

from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from collections import defaultdict

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMaskProcessor
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.utils import get_prunable_layers_in_out_channels
from nncf.common.pruning.utils import get_rounded_pruned_element_number
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.pruning.utils import get_next_nodes_of_types
from nncf.common.pruning.utils import get_output_channels
from nncf.common.pruning.utils import get_input_masks


class ShapePruninigProcessor:
    def __init__(self,
                 graph: NNCFGraph,
                 prunable_types: List[str],
                 pruning_operations_metatype: List[str],
                 pruning_groups: List[Cluster[PrunedLayerInfoBase]]):
        self._graph = graph
        self._pruning_groups = pruning_groups
        self._prunable_types = prunable_types
        self._pruning_operations_metatype = pruning_operations_metatype
        self._full_inp_channels, self._full_out_channels = get_prunable_layers_in_out_channels(graph)
        self._pruning_groups_next_nodes = self._get_cluster_next_nodes()

    @property
    def full_input_channels(self):
        return self._full_inp_channels.copy()

    @property
    def full_output_channels(self):
        return self._full_out_channels.copy()

    def calculate_in_out_channels_by_masks(self,
                                           num_of_sparse_elements_by_node: Dict[NNCFNodeName, int]) -> \
        Tuple[Dict[str, int], Dict[str, int]]:
        """
        Imitates filters pruning by removing output filters zeroed by pruning masks in each pruning group
        and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

        :param num_of_sparse_elements_by_node: A dictionary of num_of_sparse_elements of each pruning node.
        :return Dictionary of new input channels number {node_name: channels_num}
        """
        def get_num_of_sparse_elements_by_node(node_name: str) -> int:
            return num_of_sparse_elements_by_node[node_name]

        return self._calculate_in_out_channels(get_num_of_sparse_elements_by_node)
    
    def calculate_in_out_channels_in_uniformly_pruned_model(self, pruning_level: float) -> \
        Tuple[Dict[str, int], Dict[str, int]]:
        """
        Imitates filters pruning by removing `pruning_rate` percent of output filters in each pruning group
        and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

        :param pruning_level: Target pruning rate.
        :return Tuple of dictionarise of new input and output channels number {node_name: channels_num}
        """

        def get_num_of_sparse_elements_by_node(node_name: str) -> int:
            old_out_channels = self._full_out_channels[node_name]
            return get_rounded_pruned_element_number(old_out_channels, pruning_level)

        return self._calculate_in_out_channels(get_num_of_sparse_elements_by_node)


    def prune_cluster_shapes(self, cluster: Union[int, Cluster],
                             pruned_elems: int,
                             input_channels: Dict[NNCFNodeName, int],
                             output_channels: Dict[NNCFNodeName, int]) -> None:
        assert isinstance(cluster, (int, Cluster)), 'Wrong type for cluster param'
        if isinstance(cluster, int):
            cluster = self._pruning_groups.get_cluster_by_id(cluster)

        if not pruned_elems:
            return

        for node in cluster.elements:
            output_channels[node.node_name] -= pruned_elems
            if node.is_depthwise:
                input_channels[node.node_name] -= pruned_elems

        # Prune in channels in all next nodes
        next_nodes = self._pruning_groups_next_nodes[cluster.id]
        for next_node in next_nodes:
            input_channels[next_node.node_name] -= pruned_elems * next_node.sparse_multiplier

    def _calculate_in_out_channels(self, sparse_elements_counter: Callable[[str], int]) -> \
        Tuple[Dict[str, int], Dict[str, int]]:
        tmp_in_channels = self._full_inp_channels.copy()
        tmp_out_channels = self._full_out_channels.copy()

        for group in self._pruning_groups.get_all_clusters():
            layer_name = group.elements[0].node_name
            assert all(tmp_out_channels[layer_name] == tmp_out_channels[node.node_name] for node in
                       group.elements)
            # Prune all nodes in cluster (by output channels)
            num_of_sparse_elems = sparse_elements_counter(layer_name)

            self.prune_cluster_shapes(cluster=group, pruned_elems=num_of_sparse_elems,
                                      input_channels=tmp_in_channels,
                                      output_channels=tmp_out_channels)

        return tmp_in_channels, tmp_out_channels

    def get_prunable_layers_in_out_channels(self):
        return self._full_inp_channels.copy(), self._full_out_channels.copy()

    class NextNode:
        def __init__(self, node_name, sparse_multiplier):
            self.node_name = node_name
            self.sparse_multiplier = sparse_multiplier

    def _get_next_node_sparse_multiplier(self, next_node, cluster):
        for input_mask in get_input_masks(next_node, self._graph):
            if not input_mask:
                continue
            mask_producers = input_mask.mask_producers
            for cluster_node in cluster.elements:
                if cluster_node.nncf_node_id in mask_producers:
                    return mask_producers[cluster_node.nncf_node_id].sparse_multiplier

        raise RuntimeError('Next node for cluster {cluster} doesn\'t have closing mask')
    
    def _get_cluster_next_nodes(self) -> Dict[int, List['NextNode']]:
        """
        Finds nodes of `prunable_types` types that receive the output of a pruned cluster as input
        and collects all info specified in NextNode.

        :return Dictionary of next nodes by cluster {cluster_id: [node]}.
        """
        for pruned_layer_info in self._pruning_groups.get_all_nodes():
            node = self._graph.get_node_by_id(pruned_layer_info.nncf_node_id)
            node.data['output_mask'] = SymbolicMask(get_output_channels(node), node.node_id)

        MaskPropagationAlgorithm(self._graph,
                                 self._pruning_operations_metatype,
                                 SymbolicMaskProcessor).mask_propagation()

        next_nodes = defaultdict(list) 
        for cluster in self._pruning_groups.get_all_clusters():
            next_nodes_cluster = set()
            cluster_nodes = set()
            for pruned_layer_info in cluster.elements:
                nncf_cluster_node = self._graph.get_node_by_id(pruned_layer_info.nncf_node_id)
                cluster_nodes.add(nncf_cluster_node)
                curr_next_nodes = get_next_nodes_of_types(self._graph, nncf_cluster_node, self._prunable_types)

                next_nodes_cluster = next_nodes_cluster.union(curr_next_nodes)
            next_nodes_cluster = next_nodes_cluster - cluster_nodes
            for next_node in next_nodes_cluster:
                sparse_multiplier = self._get_next_node_sparse_multiplier(next_node, cluster)
                next_nodes[cluster.id].append(self.NextNode(next_node.node_name, sparse_multiplier)) 
        return next_nodes


class WeightsFlopsCalculator:
    def __init__(self,
                 graph: NNCFGraph,
                 output_shapes: Dict[NNCFNodeName, List[int]],
                 conv_op_metatypes: List[OperatorMetatype],
                 linear_op_metatypes: List[OperatorMetatype]):
        self._graph = graph
        self._conv_op_metatypes = conv_op_metatypes
        self._linear_op_metatypes = linear_op_metatypes
        self._output_shapes = output_shapes

    def count_flops_and_weights(self,
                                input_channels: Dict[NNCFNodeName, int] = None,
                                output_channels: Dict[NNCFNodeName, int] = None,
                                kernel_sizes: Dict[NNCFNodeName, Tuple[int, int]] = None,
                                op_addresses_to_skip: List[str] = None
                                ) -> Tuple[int, int]:
        """
        Counts the number weights and FLOPs in the model for convolution and fully connected layers.

        :param input_channels: Dictionary of input channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param output_channels: Dictionary of output channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param kernel_sizes: Dictionary of kernel sizes in convolutions.
            If not specified, taken from the graph. {node_name: kernel_size}. It's only supposed to be used in NAS in case
            of Elastic Kernel enabled.
        :param op_addresses_to_skip: List of operation addresses of layers that should be skipped from calculation.
            It's only supposed to be used in NAS in case of Elastic Depth enabled.
        :return number of FLOPs for the model
                number of weights (params) in the model
        """
        flops_pers_node, weights_per_node = self.count_flops_and_weights_per_node(input_channels, output_channels,
                                                                                  kernel_sizes, op_addresses_to_skip)
        return sum(flops_pers_node.values()), sum(weights_per_node.values())

    def count_flops_and_weights_per_node(self,
                                         input_channels: Dict[NNCFNodeName, int] = None,
                                         output_channels: Dict[NNCFNodeName, int] = None,
                                         kernel_sizes: Dict[NNCFNodeName, Tuple[int, int]] = None,
                                         op_addresses_to_skip: List[NNCFNodeName] = None) -> \
        Tuple[Dict[NNCFNodeName, int], Dict[NNCFNodeName, int]]:
        """
        Counts the number weights and FLOPs per node in the model for convolution and fully connected layers.

        :param input_channels: Dictionary of input channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param output_channels: Dictionary of output channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param kernel_sizes: Dictionary of kernel sizes in convolutions.
            If not specified, taken from the graph. {node_name: kernel_size}. It's only supposed to be used in NAS in case
            of Elastic Kernel enabled.
        :param op_addresses_to_skip: List of operation addresses of layers that should be skipped from calculation.
            It's only supposed to be used in NAS in case of Elastic Depth enabled.
        :return Dictionary of FLOPs number {node_name: flops_num}
                Dictionary of weights number {node_name: weights_num}
        """
        flops = {}
        weights = {}
        input_channels = input_channels or {}
        output_channels = output_channels or {}
        kernel_sizes = kernel_sizes or {}
        op_addresses_to_skip = op_addresses_to_skip or []
        for node in self._graph.get_nodes_by_metatypes(self._conv_op_metatypes):
            name = node.node_name
            if name in op_addresses_to_skip:
                continue
            num_in_channels = input_channels.get(name, node.layer_attributes.in_channels)
            num_out_channels = output_channels.get(name, node.layer_attributes.out_channels)
            kernel_size = kernel_sizes.get(name, node.layer_attributes.kernel_size)
            if is_prunable_depthwise_conv(node):
                # Prunable depthwise conv processed in special way
                # because common way to calculate filters per
                # channel for such layer leads to zero in case
                # some of the output channels are pruned.
                filters_per_channel = 1
            else:
                filters_per_channel = num_out_channels // node.layer_attributes.groups

            flops_numpy = 2 * np.prod(kernel_size) * num_in_channels * filters_per_channel * np.prod(self._output_shapes[name])
            weights_numpy = np.prod(kernel_size) * num_in_channels * filters_per_channel
            flops[name] = flops_numpy.astype(int).item()
            weights[name] = weights_numpy.astype(int).item()

        for node in self._graph.get_nodes_by_metatypes(self._linear_op_metatypes):
            name = node.node_name
            if name in op_addresses_to_skip:
                continue

            num_in_features = input_channels.get(name, node.layer_attributes.in_features)
            num_out_features = output_channels.get(name, node.layer_attributes.out_features)

            flops_numpy = 2 * num_in_features * num_out_features * np.prod(self._output_shapes[name][:-1])
            weights_numpy = num_in_features * num_out_features 
            flops[name] = flops_numpy
            weights[name] = weights_numpy

        return flops, weights

    def count_filters_num(self,
                          output_channels: Dict[NNCFNodeName, int] = None) -> int:
        """
        Counts filters of `op_metatypes` layers taking into account new output channels number.
    
        :param output_channels:  A dictionary of output channels number in pruned model.
        :return: Current number of filters according to given graph and output channels.
        """
        filters_num = 0
        output_channels = output_channels or {}
        for node in self._graph.get_nodes_by_metatypes(self._conv_op_metatypes + self._linear_op_metatypes):
            filters_num += output_channels.get(node.node_name, get_output_channels(node))
        return filters_num
