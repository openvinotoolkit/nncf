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

from collections import defaultdict
from typing import Dict, List, Optional

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.utils import get_first_nodes_of_type
from nncf.common.logging import nncf_logger
from nncf.common.pruning.clusterization import Cluster
from nncf.common.pruning.clusterization import Clusterization
from nncf.common.pruning.mask_propagation import MaskPropagationAlgorithm
from nncf.common.pruning.model_analysis import ModelAnalyzer
from nncf.common.pruning.model_analysis import cluster_special_ops
from nncf.common.pruning.utils import PruningAnalysisDecision
from nncf.common.pruning.utils import PruningAnalysisReason
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.utils import get_output_channels
from nncf.common.pruning.utils import get_previous_convs
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.utils import is_batched_linear
from nncf.common.pruning.utils import is_conv_with_downsampling
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import is_prunable_depthwise_conv
from nncf.common.scopes import should_consider_scope


class PruningNodeSelector:
    """
    Determines which of the nodes with pruning types should be pruned
    and which of them should be pruned together.
    """

    def __init__(
        self,
        pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
        prune_operations_types: List[str],
        grouping_operations_types: List[str],
        ignored_scopes: Optional[List[str]],
        target_scopes: Optional[List[str]],
        prune_first: bool,
        prune_downsample_convs: bool,
    ):
        """
        :param pruning_operator_metatypes: registry with operation metatypes pruning algorithm is aware of, i.e.
            metatypes describing operations with common pruning mask application and propagation properties, e.g.
            IdentityMaskForwardOps unifies operations that propagate pruning masks as is (relu, swish etc.), whereas
            Convolution unifies different convolution operations (conv1d, conv2d, conv3d) which accepts some input masks
            and provide some output masks.
        :param prune_operations_types: Types of operations with prunable parameters.
        :param grouping_operations_types: Types of operations causing the need to prune connected to them
            operations together.
        :param ignored_scopes: Ignored scopes.
        :param target_scopes: Target scopes.
        :param prune_first: Whether to prune first convolution or not.
        :param prune_downsample_convs: Whether to prune downsample Convolutional layers (with stride > 1) or not.
        """
        self._pruning_operator_metatypes = pruning_operator_metatypes
        pruning_op_metatypes_dict = self._pruning_operator_metatypes.registry_dict
        self._identity_mask_propagation_op_metatype = pruning_op_metatypes_dict["identity_mask_propagation"]
        self._stop_propagation_op_metatype = pruning_op_metatypes_dict["stop_propagation_ops"]

        self._prune_operations_types = prune_operations_types
        self._grouping_operations_types = grouping_operations_types

        self._ignored_scopes = ignored_scopes
        self._target_scopes = target_scopes

        self._prune_first = prune_first
        self._prune_downsample_convs = prune_downsample_convs

    def create_pruning_groups(self, graph: NNCFGraph) -> Clusterization[NNCFNode]:
        """
        This function groups ALL nodes with pruning types to groups that should be pruned together.
            1. Create clusters for special ops (eltwises) that should be pruned together
            2. Create groups of nodes that should be pruned together (taking into account clusters of special ops)
            3. Add remaining single nodes
            4. Unite clusters for Conv + Depthwise conv (should be pruned together too)
            5. Checks for groups (all nodes in group can prune or all group can't be pruned)
        Return groups of modules that should be pruned together.

        :param graph: Graph to work with and their initialization parameters as values.
        :return: Clusterization of pruned nodes.
        """

        all_nodes_to_prune = graph.get_nodes_by_types(self._prune_operations_types)  # NNCFNodes here

        # 1. Clusters for special ops
        identity_like_types = self._identity_mask_propagation_op_metatype.get_all_op_aliases()
        special_ops_clusterization = cluster_special_ops(graph, self._grouping_operations_types, identity_like_types)

        pruned_nodes_clusterization = Clusterization[NNCFNode](lambda x: x.node_id)

        # 2. Clusters for nodes that should be pruned together (taking into account clusters for special ops)
        for i, cluster in enumerate(special_ops_clusterization.get_all_clusters()):
            all_pruned_inputs = {}
            clusters_to_merge = []

            for node in cluster.elements:
                sources = get_sources_of_node(node, graph, self._prune_operations_types)
                for source_node in sources:
                    if pruned_nodes_clusterization.is_node_in_clusterization(source_node.node_id):
                        # Merge clusters if some node already added in another cluster
                        cluster = pruned_nodes_clusterization.get_cluster_containing_element(source_node.node_id)
                        clusters_to_merge.append(cluster.id)
                    elif source_node.node_id not in all_pruned_inputs:
                        all_pruned_inputs[source_node.node_id] = source_node

            if all_pruned_inputs:
                cluster = Cluster[NNCFNode](i, all_pruned_inputs.values(), all_pruned_inputs.keys())
                clusters_to_merge.append(cluster.id)
                pruned_nodes_clusterization.add_cluster(cluster)

            # Merge clusters if one source node in several special ops clusters
            pruned_nodes_clusterization.merge_list_of_clusters(clusters_to_merge)

        last_cluster_idx = len(special_ops_clusterization.get_all_clusters())

        # 3. Add remaining single nodes as separate clusters
        for node in all_nodes_to_prune:
            if not pruned_nodes_clusterization.is_node_in_clusterization(node.node_id):
                cluster = Cluster[NNCFNode](last_cluster_idx, [node], [node.node_id])
                pruned_nodes_clusterization.add_cluster(cluster)

                last_cluster_idx += 1

        stop_propagation_ops = self._stop_propagation_op_metatype.get_all_op_aliases()
        # 4. Merge clusters for Conv + Depthwise conv (should be pruned together too)
        for node in all_nodes_to_prune:
            cluster_id = pruned_nodes_clusterization.get_cluster_containing_element(node.node_id).id

            if is_prunable_depthwise_conv(node):
                previous_convs = get_previous_convs(graph, node, self._prune_operations_types, stop_propagation_ops)
                previous_clusters = [
                    pruned_nodes_clusterization.get_cluster_containing_element(node.node_id).id
                    for node in previous_convs
                ]
                pruned_nodes_clusterization.merge_list_of_clusters([cluster_id] + previous_clusters)

        # 5. Merge nodes into one cluster if some module forwards several times
        multiforward_nodes = self._get_multiforward_nodes(graph)
        for list_of_nodes in multiforward_nodes:
            clusters_to_merge = [
                pruned_nodes_clusterization.get_cluster_containing_element(node.node_id).id for node in list_of_nodes
            ]
            pruned_nodes_clusterization.merge_list_of_clusters(clusters_to_merge)

            # Merge previous convolutions into one cluster
            all_previous_convs = []
            for node in list_of_nodes:
                nncf_node = graph.get_node_by_id(node.node_id)
                previous_convs = get_previous_convs(
                    graph, nncf_node, self._prune_operations_types, stop_propagation_ops
                )
                # Check if previous node isn't multiforward,
                # in case of multiforward nodes cycle
                for previous_conv in previous_convs:
                    if previous_conv not in list_of_nodes:
                        all_previous_convs.append(previous_conv)

            previous_clusters = [
                pruned_nodes_clusterization.get_cluster_containing_element(node.node_id).id
                for node in all_previous_convs
            ]
            pruned_nodes_clusterization.merge_list_of_clusters(previous_clusters)

        # 6. Checks for groups (all nodes in group can be pruned or all group can't be pruned).
        model_analyser = ModelAnalyzer(graph, self._pruning_operator_metatypes, self._prune_operations_types)
        can_prune_analysis = model_analyser.analyse_model_before_pruning()
        can_prune_and_should_prune_analysis = self._should_prune_groups_analysis(
            graph, pruned_nodes_clusterization, can_prune_analysis
        )
        can_prune_final_analysis = self._pruning_dimensions_analysis(
            graph, pruned_nodes_clusterization, can_prune_and_should_prune_analysis
        )
        self._filter_groups(pruned_nodes_clusterization, can_prune_final_analysis)
        return pruned_nodes_clusterization

    def _get_multiforward_nodes(self, graph: NNCFGraph) -> List[List[NNCFNode]]:
        """
        Groups nodes based on their `layer_name` property to determine groups of nodes belonging to
        a single weighted layer object in the model, i.e. the group of operations in the graph that reuse one and
        the same set of weights, and returns the groups that have more than one element.

        :return: List of lists of nodes; each list corresponds to a group of nodes united by the common
         underlying layer object of the original model.
        """
        ret = defaultdict(list)
        for node in graph.get_nodes_by_types(self._prune_operations_types):
            ret[node.layer_name].append(node)
        return [ret[module_identifier] for module_identifier in ret if len(ret[module_identifier]) > 1]

    def _pruning_dimensions_analysis(
        self,
        graph: NNCFGraph,
        pruned_nodes_clusterization: Clusterization,
        can_prune_after_check: Dict[int, PruningAnalysisDecision],
    ) -> Dict[int, PruningAnalysisDecision]:
        """
        Checks:
        1) All nodes that were marked as prunable after the model analysis and compatibility check vs.
        pruning algo have a correct correspondent closing node on each path from self to outputs;
        2) Pruning dimensions of all nodes in all cluster groups are equal.

        :param graph: Graph to work with.
        :param pruned_nodes_clusterization: Pruned nodes clusterization.
        :param can_prune_after_check: Dict of node indices vs the decision made by previous steps;
            the decision is true only for the nodes that do not conflict with mask propagation and
            are supported by the NNCF pruning algorithm
        :return: Pruning node analysis after model analyzer, pruning algo compatibility and pruning dimensions checks.
        """

        nodes_of_group_with_non_eq_pruning_dim = self._check_internal_groups_dim(pruned_nodes_clusterization)
        can_prune_after_check_updated = can_prune_after_check.copy()
        for node_id, val in nodes_of_group_with_non_eq_pruning_dim.items():
            can_prune_after_check_updated[node_id] = can_prune_after_check_updated[node_id].join(val)

        return self._check_all_closing_nodes_are_feasible(graph, can_prune_after_check_updated)

    def _check_all_closing_nodes_are_feasible(
        self, graph: NNCFGraph, can_prune_after_check: Dict[int, PruningAnalysisDecision]
    ) -> Dict[int, PruningAnalysisDecision]:
        """
        Check all nodes that were marked as prunable after the model analysis and compatibility check vs.
        pruning algo have a correct correspondent closing node on each path from self to outputs.

        :param graph: Graph to work with.
        :param can_prune_after_check: Dict of node indices vs the decision made by previous steps;
            the decision is true only for the nodes that do not conflict with mask propagation and
            are supported by the NNCF pruning algorithm
        :return: Pruning node analysis after model analyzer, pruning algo compatibility and pruning dimensions checks.
        """
        mask_prop_algo = MaskPropagationAlgorithm(graph, self._pruning_operator_metatypes)
        can_prune_by_dim = mask_prop_algo.symbolic_mask_propagation(self._prune_operations_types, can_prune_after_check)

        can_prune_for_prunable_layers = {
            node_id: can_prune_after_check[node_id].join(can_prune_by_dim[node_id]) for node_id in can_prune_by_dim
        }

        can_prune_updated = can_prune_after_check.copy()
        can_prune_updated.update(can_prune_for_prunable_layers)
        return can_prune_updated

    def _check_internal_groups_dim(
        self, pruned_nodes_clusterization: Clusterization
    ) -> Dict[int, PruningAnalysisDecision]:
        """
        Checks pruning dimensions of all nodes in each cluster group are equal and
        returns nodes of clusters that failed the check.

        :param pruned_nodes_clusterization: Pruned nodes clusterization.
        :returns: Pruning analysis decisions for nodes which have
            not equal pruning dimensions in a cluster they are belong to.
        """
        retval = {}
        for cluster in pruned_nodes_clusterization.get_all_clusters():
            has_equal_amount_of_channel = all(
                get_output_channels(cluster.elements[0]) == get_output_channels(node) for node in cluster.elements[1:]
            )
            if not has_equal_amount_of_channel:
                retval.update(
                    {
                        node.node_id: PruningAnalysisDecision(False, PruningAnalysisReason.INCOMPATIBLE_DIMS_IN_CLUSTER)
                        for node in cluster.elements
                    }
                )
        return retval

    def _should_prune_groups_analysis(
        self,
        graph: NNCFGraph,
        pruned_nodes_clusterization: Clusterization,
        can_prune: Dict[int, PruningAnalysisDecision],
    ) -> Dict[int, PruningAnalysisDecision]:
        """
        Check whether all nodes in group can be pruned based on user-defined constraints and
        model analysis. Otherwise the whole group cannot be pruned.

        :param graph: Graph to work with.
        :param pruned_nodes_clusterization: Clusterization with pruning nodes groups.
        :param can_prune: Complete pruning analysis about each graph node.
        :return:
        """
        should_prune = {}
        for cluster in pruned_nodes_clusterization.get_all_clusters():
            should_prune_decisions = [self._is_module_prunable(graph, node) for node in cluster.elements]
            can_prune_decisions = [can_prune[node.node_id] for node in cluster.elements]
            decisions = [can.join(should) for can, should in zip(can_prune_decisions, should_prune_decisions)]
            if not all(decisions):
                updated_decisions = {}
                for node, decision in zip(cluster.elements, decisions):
                    if decision:
                        updated_decisions[node.node_id] = PruningAnalysisDecision(
                            False, PruningAnalysisReason.IN_GROUP_OF_UNPRUNABLE
                        )
                    else:
                        updated_decisions[node.node_id] = decision

                should_prune.update(updated_decisions)

        can_prune_updated = can_prune.copy()
        can_prune_updated.update(should_prune)
        return can_prune_updated

    def _filter_groups(
        self, pruned_nodes_clusterization: Clusterization, can_prune: Dict[int, PruningAnalysisDecision]
    ) -> None:
        """
        Check whether all nodes in group can be pruned based on user-defined constraints and
        connections inside the network. Otherwise the whole group cannot be pruned and will be
        removed the clusterization.

        :param pruned_nodes_clusterization: Clusterization with pruning nodes groups.
        :param can_prune: Can this node be pruned or not.
        """
        for cluster in pruned_nodes_clusterization.get_all_clusters():
            nodes_decisions = [can_prune[node.node_id] for node in cluster.elements]
            nodes_names = [node.node_name for node in cluster.elements]
            if not all(nodes_decisions):
                cannot_prune_messages = []
                for name, decision in zip(nodes_names, nodes_decisions):
                    if not decision:
                        message = PruningAnalysisReason.message(name, decision)
                        if message:
                            cannot_prune_messages.append(message)

                nncf_logger.debug(
                    f'Could not prune node group [{", ".join(nodes_names)}], '
                    f'reason: {", ".join(cannot_prune_messages)}.'
                )
                pruned_nodes_clusterization.delete_cluster(cluster.id)
            else:
                nncf_logger.debug(f'Node group [{", ".join(nodes_names)}] will be pruned together.')

    def _is_module_prunable(self, graph: NNCFGraph, node: NNCFNode) -> PruningAnalysisDecision:
        """
        Check whether we should prune module corresponding to provided node
        according to algorithm parameters.

        :param graph: Graph to work with.
        :param node: Node to check.
        :return: Pruning analysis decision.
        """
        stop_propagation_ops = self._stop_propagation_op_metatype.get_all_op_aliases()
        types_to_track = self._prune_operations_types + stop_propagation_ops
        input_non_pruned_nodes = get_first_nodes_of_type(graph, types_to_track)
        node_name = node.node_name

        if not should_consider_scope(node_name, self._ignored_scopes, self._target_scopes):
            return PruningAnalysisDecision(False, PruningAnalysisReason.IGNORED_SCOPE)

        if not self._prune_first and node in input_non_pruned_nodes:
            return PruningAnalysisDecision(False, PruningAnalysisReason.FIRST_CONV)

        if is_grouped_conv(node) and not is_prunable_depthwise_conv(node):
            return PruningAnalysisDecision(False, PruningAnalysisReason.GROUP_CONV)

        if not self._prune_downsample_convs and is_conv_with_downsampling(node):
            return PruningAnalysisDecision(False, PruningAnalysisReason.DOWNSAMPLE_CONV)

        if is_batched_linear(node, graph):
            return PruningAnalysisDecision(False, PruningAnalysisReason.BATCHED_LINEAR)

        return PruningAnalysisDecision(True)
