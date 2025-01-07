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

from typing import Dict, List

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.operations import BasePruningOp
from nncf.common.pruning.utils import PruningAnalysisDecision
from nncf.common.pruning.utils import PruningAnalysisReason
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import find_next_nodes_not_of_types
from nncf.common.pruning.utils import is_prunable_depthwise_conv


def get_position(nodes_list: List[NNCFNode], idx: int):
    for i, node in enumerate(nodes_list):
        if node.node_id == idx:
            return i
    return None


def merge_clusters_for_nodes(nodes_to_merge: List[NNCFNode], clusterization: Clusterization):
    """
    Merges clusters to which nodes from nodes_to_merge belongs.

    :param nodes_to_merge: All nodes are clusters for which should be merged.
    :param clusterization: Clusterization of nodes to work with.
    """
    if len(nodes_to_merge) <= 1:
        return

    # Will merge cluster with highest importance with others pairwise
    max_importance_node_id = None
    max_importance = 0
    for node in nodes_to_merge:
        importance = clusterization.get_cluster_containing_element(node.node_id).importance
        if importance > max_importance:
            max_importance_node_id = node.node_id
            max_importance = importance

    max_importance_cluster_id = clusterization.get_cluster_containing_element(max_importance_node_id).id
    for node in nodes_to_merge:
        if node.node_id != max_importance_node_id:
            current_node_cluster_id = clusterization.get_cluster_containing_element(node.node_id).id
            if current_node_cluster_id != max_importance_cluster_id:
                clusterization.merge_clusters(max_importance_cluster_id, current_node_cluster_id)


def cluster_special_ops(
    graph: NNCFGraph, special_types: List[str], identity_types: List[str]
) -> Clusterization[NNCFNode]:
    """
    This model will cluster all operations with type from special_types. Connected nodes is nodes that:
        1. Have path between nodes with only identity type nodes on it
        2. Have common input (identity type nodes can be on path from this input)

    :param graph: Graph to work with.
    :param special_types: List of types that should be grouped to groups of dependent nodes.
    :return: Clusterization of `special_types` nodes to the dependent groups.
    """
    topologically_sorted_nodes = graph.topological_sort()
    all_special_nodes = [node for node in graph.get_all_nodes() if node.node_type in special_types]

    # 0. Initially all nodes is a separate clusters
    clusterization = Clusterization[NNCFNode](lambda x: x.node_id)
    for i, node in enumerate(all_special_nodes):
        cluster = Cluster[NNCFNode](i, [node], [get_position(topologically_sorted_nodes, node.node_id)])
        clusterization.add_cluster(cluster)

    for node in topologically_sorted_nodes:
        if node.node_type in identity_types:
            continue

        all_outputs = find_next_nodes_not_of_types(graph, node, identity_types)
        all_output_special_nodes = [node for node in all_outputs if node.node_type in special_types]
        if node.node_type in special_types:
            all_output_special_nodes.append(node)
        merge_clusters_for_nodes(all_output_special_nodes, clusterization)

    return clusterization


class ModelAnalyzer:
    """
    Analyze the model before pruning to understand which parts could potentially be pruned without conflicts
    (all nodes that can't get pruned input will receive a non-pruned input).

    The algorithm consists of three steps:
        1. Set attribute `accept_pruned_input` to all nodes. This attribute shows whether this node can
        potentially get pruned input or node.
        2. Calculate `can_prune` attribute for all nodes by propagating `accept_pruned_input` up
        (from the result of the network to the inputs). Node can be pruned if all outputs of this node accept
        pruned input and all outputs can be pruned.
        3. Propagates `can_prune` down from input nodes to the outputs.

    As a result, all nodes are marked by the `can_prune` attribute as potentially prunable or not.
    """

    def __init__(
        self,
        graph: NNCFGraph,
        pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
        prune_operations_types: List[str],
    ):
        """
        :param pruning_operator_metatypes: registry with operation metatypes pruning algorithm is aware of, i.e.
        metatypes describing operations with common pruning mask application and propagation properties, e.g.
        IdentityMaskForwardOps unifies operations that propagate pruning masks as is (relu, swish etc.), whereas
        Convolution unifies different convolution operations (conv1d, conv2d, conv3d) which accepts some input masks
        and provide some output masks.
        :param prune_operations_types: Types of operations with prunable parameters.
        """
        self.graph = graph
        self._pruning_operator_metatypes = pruning_operator_metatypes
        self._prune_operations_types = prune_operations_types
        pruning_op_metatypes_dict = self._pruning_operator_metatypes.registry_dict
        self._stop_propagation_op_metatype = pruning_op_metatypes_dict["stop_propagation_ops"]
        self._concat_op_metatype = pruning_op_metatypes_dict["concat"]

        self.can_prune = {idx: True for idx in self.graph.get_all_node_ids()}
        self.accept_pruned_input = {idx: True for idx in self.graph.get_all_node_ids()}

    def node_propagate_can_prune_attr(self, nncf_node: NNCFNode) -> bool:
        """
        Whether the node can propagate the `can_prune` attr. That means a node can propagate pruning mask
        (for example, activations propagate mask, but convolutions stop mask propagation).

        :param nncf_node: Node to work with.
        :return: Propagates this node can_prune throw or not.
        """
        node_type = nncf_node.node_type
        is_prunable = node_type in self._prune_operations_types
        return not is_prunable or (is_prunable and is_prunable_depthwise_conv(nncf_node))

    def node_accept_different_inputs(self, nncf_node: NNCFNode) -> bool:
        """
        Returns whether node accepts pruned and not pruned inputs as inputs at the same time.

        :return: Whether node accepts pruned and not pruned inputs as inputs at the same time.
        """
        return nncf_node.node_type in self._concat_op_metatype.get_all_op_aliases()

    def get_meta_operation_by_type_name(self, type_name: str) -> BasePruningOp:
        """
        Returns class of metaop that corresponds to `type_name` type.

        :return: Class of metaop that corresponds to `type_name` type.
        """
        cls = self._pruning_operator_metatypes.get_operator_metatype_by_op_name(type_name)
        if cls is None:
            cls = self._stop_propagation_op_metatype
        return cls

    def propagate_can_prune_attr_up(self):
        """
        Propagating can_prune attribute in reversed topological order.
        This attribute depends on accept_pruned_input and can_prune attributes of output nodes.
        Node can_prune is True if all outputs accept_pruned_input is True and all outputs
        (except convs because conv can be pruned by input and output independently).
        """
        reversed_sorted_nodes = reversed(self.graph.topological_sort())
        for node in reversed_sorted_nodes:
            # Check all output nodes accept_pruned_input attribute
            out_nodes = self.graph.get_next_nodes(node)
            outputs_accept_pruned_input = all(self.accept_pruned_input[node.node_id] for node in out_nodes)

            # Check all output nodes can_prune attribute
            outputs_will_be_pruned = all(
                self.can_prune[node.node_id] for node in out_nodes if self.node_propagate_can_prune_attr(node)
            )
            self.can_prune[node.node_id] = outputs_accept_pruned_input and outputs_will_be_pruned

    def propagate_can_prune_attr_down(self):
        """
        Propagating can_prune attribute down to fix all branching cases with one pruned and one not pruned
        branches.
        """
        sorted_nodes = self.graph.topological_sort()
        for node in sorted_nodes:
            # Propagate attribute only in not conv case
            if self.node_propagate_can_prune_attr(node):
                in_nodes = self.graph.get_previous_nodes(node)
                can_prune = all(self.can_prune[node.node_id] for node in in_nodes)
                can_prune_any = any(self.can_prune[node.node_id] for node in in_nodes)

                if (not self.node_accept_different_inputs(node) and not can_prune) or (
                    self.node_accept_different_inputs(node) and not can_prune_any
                ):
                    self.can_prune[node.node_id] = can_prune

    def set_accept_pruned_input_attr(self):
        for nncf_node in self.graph.get_all_nodes():
            cls = self.get_meta_operation_by_type_name(nncf_node.node_type)
            self.accept_pruned_input[nncf_node.node_id] = cls.accept_pruned_input(nncf_node)

    def analyse_model_before_pruning(self) -> Dict[int, PruningAnalysisDecision]:
        self.set_accept_pruned_input_attr()
        self.propagate_can_prune_attr_up()
        self.propagate_can_prune_attr_down()
        return {k: PruningAnalysisDecision(v, PruningAnalysisReason.MODEL_ANALYSIS) for k, v in self.can_prune.items()}
