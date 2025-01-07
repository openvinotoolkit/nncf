# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Dict, List, Tuple

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFNodeName
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.structs import PrunedLayerInfoBase
from nncf.common.pruning.symbolic_mask import SymbolicMask
from nncf.common.pruning.symbolic_mask import SymbolicMaskProcessor
from nncf.common.pruning.utils import get_input_masks
from nncf.common.pruning.utils import get_next_nodes_of_types
from nncf.common.pruning.utils import get_output_channels
from nncf.common.pruning.utils import get_prunable_layers_in_out_channels
from nncf.common.pruning.utils import get_rounded_pruned_element_number


class ShapePruningProcessor:
    """
    Collection of shape pruning functions. Class instance keeps
    only parameters that are constant during
    compression algorithms execution.
    """

    def __init__(self, prunable_types: List[str], pruning_operations_metatype: List[str]):
        """
        Constructor.

        :param prunable_types: Types of nodes that will be returned.
        :param pruning_operations_metatype: Metatypes of nodes that will be returned.
        """
        self._prunable_types = prunable_types
        self._pruning_operations_metatype = pruning_operations_metatype

    def calculate_in_out_channels_by_masks(
        self,
        graph: NNCFGraph,
        pruning_groups: List[Cluster[PrunedLayerInfoBase]],
        pruning_groups_next_nodes: Dict[int, List[Dict[str, Any]]],
        num_of_sparse_elements_by_node: Dict[NNCFNodeName, int],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Imitates filters pruning by removing output filters zeroed by pruning masks in each pruning group
        and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

        :param graph: NNCFGraph.
        :param pruning_groups: A list of pruning groups.
        :param pruning_groups_next_nodes: A dictionary of next nodes of each pruning group.
        :param num_of_sparse_elements_by_node: A dictionary of num_of_sparse_elements of each pruning node.
        :return Dictionary of new input channels number {node_name: channels_num}
        """

        def get_sparser(full_output_channels):
            def get_num_of_sparse_elements_by_node(node_name: str) -> int:
                return num_of_sparse_elements_by_node[node_name]

            return get_num_of_sparse_elements_by_node

        return self._calculate_in_out_channels(get_sparser, graph, pruning_groups, pruning_groups_next_nodes)

    def calculate_in_out_channels_in_uniformly_pruned_model(
        self,
        graph: NNCFGraph,
        pruning_groups: List[Cluster[PrunedLayerInfoBase]],
        pruning_groups_next_nodes: Dict[int, List[Dict[str, Any]]],
        pruning_level: float,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Imitates filters pruning by removing `pruning_level` percent of output filters in each pruning group
        and updating corresponding input channels number in `pruning_groups_next_nodes` nodes.

        :param graph: NNCFGraph.
        :param pruning_groups: A list of pruning groups.
        :param pruning_groups_next_nodes: A dictionary of next nodes of each pruning group.
        :param pruning_level: Target pruning level.
        :return Tuple of dictionarise of new input and output channels number {node_name: channels_num}
        """

        def get_sparser(full_output_channels):
            def get_num_of_sparse_elements_by_node(node_name: str) -> int:
                old_out_channels = full_output_channels[node_name]
                return get_rounded_pruned_element_number(old_out_channels, pruning_level)

            return get_num_of_sparse_elements_by_node

        return self._calculate_in_out_channels(get_sparser, graph, pruning_groups, pruning_groups_next_nodes)

    def prune_cluster_shapes(
        self,
        cluster: Cluster[PrunedLayerInfoBase],
        pruned_elems: int,
        pruning_groups_next_nodes: Dict[int, List[Dict[str, Any]]],
        input_channels: Dict[NNCFNodeName, int],
        output_channels: Dict[NNCFNodeName, int],
    ) -> None:
        """
        Imitates filter pruning by removing `pruned_elems` elements from
        input/output channels corresponded to feeded cluster.

        :param cluster:  A PrunedLayerInfoBase cluster.
        :param pruned_elems: Amount of channels/elements to prune.
        :param pruning_groups_next_nodes: A dictionary of next nodes of each pruning group.
        :param input_channels: A dictionary of input channels number in prunable layers.
            Will be modified according to the filter pruning algorithm.
        :param output_channels: A dictionary of output channels number in prunable layers.
            Will be modified according to the filter pruning algorithm.
        """
        if not pruned_elems:
            return

        for node in cluster.elements:
            output_channels[node.node_name] -= pruned_elems
            if node.is_depthwise:
                input_channels[node.node_name] -= pruned_elems

        # Prune in channels in all next nodes
        next_nodes_info = pruning_groups_next_nodes[cluster.id]
        for next_node_info in next_nodes_info:
            input_channels[next_node_info["node_name"]] -= pruned_elems * next_node_info["sparse_multiplier"]

    def _calculate_in_out_channels(
        self,
        sparse_elements_counter_getter: Callable[[Dict[NNCFNodeName, int]], Callable[[NNCFNodeName], int]],
        graph: NNCFGraph,
        pruning_groups: List[Cluster[PrunedLayerInfoBase]],
        pruning_groups_next_nodes: Dict[int, List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        full_input_channels, full_output_channels = get_prunable_layers_in_out_channels(graph)
        tmp_in_channels, tmp_out_channels = full_input_channels.copy(), full_output_channels.copy()
        sparse_elements_counter = sparse_elements_counter_getter(full_output_channels)

        for group in pruning_groups.get_all_clusters():
            layer_name = group.elements[0].node_name
            assert all(tmp_out_channels[layer_name] == tmp_out_channels[node.node_name] for node in group.elements)
            # Prune all nodes in cluster (by output channels)
            num_of_sparse_elems = sparse_elements_counter(layer_name)

            self.prune_cluster_shapes(
                cluster=group,
                pruned_elems=num_of_sparse_elems,
                input_channels=tmp_in_channels,
                output_channels=tmp_out_channels,
                pruning_groups_next_nodes=pruning_groups_next_nodes,
            )

        return tmp_in_channels, tmp_out_channels

    def _get_next_node_sparse_multiplier(
        self, graph: NNCFGraph, next_node: NNCFNode, cluster: Clusterization[PrunedLayerInfoBase]
    ) -> int:
        cluster_nodes_idxs = {node.nncf_node_id for node in cluster.elements}
        for input_mask in get_input_masks(next_node, graph):
            if not input_mask:
                continue
            for mask_producer in input_mask.mask_producers:
                if mask_producer.id in cluster_nodes_idxs:
                    return mask_producer.sparse_multiplier

        raise nncf.ValidationError(f"Next node for cluster {cluster.elements} doesn't have closing mask")

    def get_next_nodes(
        self, graph: NNCFGraph, pruning_groups: Clusterization[PrunedLayerInfoBase]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Finds nodes of `prunable_types` types that receive the output of a pruned cluster as input
        and collects all info specified in NextNode.

        :param graph: NNCFGraph.
        :param pruning_groups: `Clusterization` of pruning groups.
        :return Dictionary of next nodes by cluster {cluster_id: [node]}.
        """
        # 1. Propagate symbolic masks throught the net
        for pruned_layer_info in pruning_groups.get_all_nodes():
            node = graph.get_node_by_id(pruned_layer_info.nncf_node_id)
            node.attributes["output_mask"] = SymbolicMask(get_output_channels(node), node.node_id)

        MaskPropagationAlgorithm(graph, self._pruning_operations_metatype, SymbolicMaskProcessor).mask_propagation()

        # 2. Find next nodes and correspondent sparse multipliers
        next_nodes = {}
        for cluster in pruning_groups.get_all_clusters():
            next_nodes_cluster = set()
            cluster_nodes = set()
            for pruned_layer_info in cluster.elements:
                nncf_cluster_node = graph.get_node_by_id(pruned_layer_info.nncf_node_id)
                cluster_nodes.add(nncf_cluster_node)
                curr_next_nodes = get_next_nodes_of_types(graph, nncf_cluster_node, self._prunable_types)

                next_nodes_cluster = next_nodes_cluster.union(curr_next_nodes)
            next_nodes_cluster = next_nodes_cluster - cluster_nodes
            next_nodes[cluster.id] = []
            for next_node in next_nodes_cluster:
                sparse_multiplier = self._get_next_node_sparse_multiplier(graph, next_node, cluster)
                next_nodes[cluster.id].append(
                    {"node_name": next_node.node_name, "sparse_multiplier": sparse_multiplier}
                )

        # 3. Clean graph output shapes
        for node in graph.get_all_nodes():
            node.attributes["output_shape"] = None

        return next_nodes
