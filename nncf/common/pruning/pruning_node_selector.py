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

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from collections import defaultdict

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.pruning.utils import get_sources_of_node
from nncf.common.pruning.utils import get_first_nodes_of_type
from nncf.common.pruning.utils import get_last_nodes_of_type
from nncf.common.pruning.utils import get_previous_conv
from nncf.common.pruning.utils import is_grouped_conv
from nncf.common.pruning.utils import PruningOperationsMetatypeRegistry
from nncf.common.pruning.model_analysis import ModelAnalyzer
from nncf.common.pruning.model_analysis import Clusterization
from nncf.common.pruning.model_analysis import cluster_special_ops
from nncf.common.pruning.model_analysis import NodesCluster
from nncf.common.utils.logger import logger as nncf_logger
from nncf.common.utils.backend import __nncf_backend__

if __nncf_backend__ == 'Torch':
    from nncf.pruning.utils import is_depthwise_conv
    from nncf.pruning.utils import is_conv_with_downsampling
    from nncf.graph.graph import get_module_identifier
    from nncf.utils import should_consider_scope
elif __nncf_backend__ == 'TensorFlow':
    from beta.nncf.tensorflow.pruning.utils import is_depthwise_conv
    from beta.nncf.tensorflow.pruning.utils import is_conv_with_downsampling
    from beta.nncf.tensorflow.graph.utils import get_layer_identifier as get_module_identifier
    from beta.nncf.tensorflow.utils.scopes_handle import should_consider_scope


class PruningNodeSelector:
    """
    Determines which of the nodes with pruning types should be pruned
    and which of them should be pruned together.
    """

    def __init__(self,
                 pruning_operator_metatypes: PruningOperationsMetatypeRegistry,
                 prune_operations: List[str],
                 grouping_operations: List[str],
                 ignored_scopes: Optional[List[str]],
                 target_scopes: Optional[List[str]],
                 prune_first: bool,
                 prune_last: bool,
                 prune_downsample_convs: bool):
        """
        :param pruning_operator_metatypes: registry with operation metatypes pruning algorithm is aware of, i.e.
            metatypes describing operations with common pruning mask application and propagation properties, e.g.
            IdentityMaskForwardOps unifies operations that propagate pruning masks as is (relu, swish etc.), whereas
            Convolution unifies different convolution operations (conv1d, conv2d, conv3d) which accepts some input masks
            and provide some output masks.
        :param prune_operations: Names of operations with prunable filters.
        :param grouping_operations: Names of operations causing the need to prune connected to them operations together.
        :param ignored_scopes: Ignored scopes.
        :param target_scopes: Target scopes.
        :param prune_first: Whether to prune first convolution or not.
        :param prune_last: Whether to prune last convolution or not.
        :param prune_downsample_convs: Whether to prune downsample Convolutional layers (with stride > 1) or not.
        """
        self._pruning_operator_metatypes = pruning_operator_metatypes
        pruning_op_metatypes_dict = self._pruning_operator_metatypes.registry_dict
        self._identity_mask_propagation_op_metatype = pruning_op_metatypes_dict['identity_mask_propagation']
        self._stop_propagation_op_metatype = pruning_op_metatypes_dict['stop_propagation_ops']
        self._convolution_op_metatype = pruning_op_metatypes_dict['convolution']

        self._prune_operations = prune_operations
        self._grouping_operations = grouping_operations

        self._ignored_scopes = ignored_scopes
        self._target_scopes = target_scopes

        self._prune_first = prune_first
        self._prune_last = prune_last
        self._prune_downsample_convs = prune_downsample_convs

    def create_pruning_groups(self, graph: NNCFGraph) -> Clusterization:
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
        all_nodes_to_prune = graph.get_nodes_by_types(self._prune_operations)  # NNCFNodes here

        # 1. Clusters for special ops
        identity_like_types = self._identity_mask_propagation_op_metatype.get_all_op_aliases()
        special_ops_clusterization = cluster_special_ops(graph, self._grouping_operations,
                                                         identity_like_types)

        pruned_nodes_clusterization = Clusterization("node_id")

        # 2. Clusters for nodes that should be pruned together (taking into account clusters for special ops)
        for i, cluster in enumerate(special_ops_clusterization.get_all_clusters()):
            all_pruned_inputs = []
            pruned_inputs_idxs = set()
            clusters_to_merge = list()

            for node in cluster.nodes:
                sources = get_sources_of_node(node, graph, self._prune_operations)
                for source_node in sources:
                    if pruned_nodes_clusterization.is_node_in_clusterization(source_node.node_id):
                        # Merge clusters if some node already added in another cluster
                        cluster = pruned_nodes_clusterization.get_cluster_by_node_id(source_node.node_id)
                        clusters_to_merge.append(cluster.id)
                    elif source_node.node_id not in pruned_inputs_idxs:
                        all_pruned_inputs.append(source_node)
                        pruned_inputs_idxs.add(source_node.node_id)

            if all_pruned_inputs:
                cluster = NodesCluster(i, all_pruned_inputs, [n.node_id for n in all_pruned_inputs])
                clusters_to_merge.append(cluster.id)
                pruned_nodes_clusterization.add_cluster(cluster)

            # Merge clusters if one source node in several special ops clusters
            pruned_nodes_clusterization.merge_list_of_clusters(clusters_to_merge)

        last_cluster_idx = len(special_ops_clusterization.get_all_clusters())

        # 3. Add remaining single nodes as separate clusters
        for node in all_nodes_to_prune:
            if not pruned_nodes_clusterization.is_node_in_clusterization(node.node_id):
                cluster = NodesCluster(last_cluster_idx, [node], [node.node_id])
                pruned_nodes_clusterization.add_cluster(cluster)

                last_cluster_idx += 1

        stop_propagation_ops = self._stop_propagation_op_metatype.get_all_op_aliases()
        # 4. Merge clusters for Conv + Depthwise conv (should be pruned together too)
        for node in all_nodes_to_prune:
            cluster_id = pruned_nodes_clusterization.get_cluster_by_node_id(node.node_id).id

            if is_depthwise_conv(node):
                previous_conv = get_previous_conv(graph, node, self._prune_operations, stop_propagation_ops)
                if previous_conv:
                    previous_conv_cluster_id = pruned_nodes_clusterization.get_cluster_by_node_id(
                        previous_conv.node_id).id
                    pruned_nodes_clusterization.merge_clusters(cluster_id, previous_conv_cluster_id)

        # 5. Merge nodes into one cluster if some module forwards several times
        multiforward_nodes = self._get_multiforward_nodes(graph)
        for list_of_nodes in multiforward_nodes:
            clusters_to_merge = [pruned_nodes_clusterization.get_cluster_by_node_id(id).id for id in list_of_nodes]
            pruned_nodes_clusterization.merge_list_of_clusters(clusters_to_merge)

            # Merge previous convolutions into one cluster
            previous_convs = []
            for id_ in list_of_nodes:
                nncf_node = graph.get_node_by_id(id_)
                previous_conv = get_previous_conv(graph, nncf_node, self._prune_operations, stop_propagation_ops)
                previous_convs.append(previous_conv)
            previous_clusters = [
                pruned_nodes_clusterization.get_cluster_by_node_id(node.node_id).id
                for node in previous_convs
            ]
            pruned_nodes_clusterization.merge_list_of_clusters(previous_clusters)

        # 6. Checks for groups (all nodes in group can be pruned or all group can't be pruned).
        model_analyser = ModelAnalyzer(graph, self._pruning_operator_metatypes, is_depthwise_conv)
        can_prune_analysis = model_analyser.analyse_model_before_pruning()
        self._check_pruning_groups(graph, pruned_nodes_clusterization, can_prune_analysis)
        return pruned_nodes_clusterization

    def _get_multiforward_nodes(self, graph: NNCFGraph):
        """
        Found all nodes that forward several times.

        :return: List of nodes that forward several times.
        """
        ret = defaultdict(list)
        for node in graph.get_nodes_by_types(self._prune_operations):
            module_identifier = get_module_identifier(node)
            ret[module_identifier].append(node.node_id)
        return [ret[module_identifier] for module_identifier in ret if len(ret[module_identifier]) > 1]

    def _check_pruning_groups(self, graph: NNCFGraph, pruned_nodes_clusterization: Clusterization,
                              can_prune: Dict[str, bool]):
        """
        Check whether all nodes in group can be pruned based on user-defined constraints and
        connections inside the network. Otherwise the whole group cannot be pruned.

        :param graph: Graph to work with.
        :param pruned_nodes_clusterization: Clusterization with pruning nodes groups.
        :param can_prune: Can this node be pruned or not.
        """
        for cluster in pruned_nodes_clusterization.get_all_clusters():
            cluster_nodes_names = [get_module_identifier(n) for n in cluster.nodes]
            # Check whether this node should be pruned according to the user-defined algorithm constraints
            should_prune_nodes = [self._is_module_prunable(graph, node) for node in cluster.nodes]

            # Check whether this node can be potentially pruned from architecture point of view
            can_prune_nodes = [can_prune[node.node_id] for node in cluster.nodes]
            if not all(can_prune[0] for can_prune in should_prune_nodes):
                shouldnt_prune_msgs = [should_prune[1] for should_prune in should_prune_nodes if not should_prune[0]]
                nncf_logger.info("Group of nodes [{}] can't be pruned, because some nodes should't be pruned, "
                                 "error messages for this nodes: {}".format(", ".join(cluster_nodes_names),
                                                                            ", ".join(shouldnt_prune_msgs)))
                pruned_nodes_clusterization.delete_cluster(cluster.id)
            elif not all(can_prune_nodes):
                cant_prune_nodes_names = [get_module_identifier(node) for node in cluster.nodes
                                          if not can_prune[node.node_id]]
                nncf_logger.info("Group of nodes [{}] can't be pruned, because {} nodes can't be pruned "
                                 "according to model analysis"
                                 .format(", ".join(cluster_nodes_names), ", ".join(cant_prune_nodes_names)))
                pruned_nodes_clusterization.delete_cluster(cluster.id)
            else:
                nncf_logger.info("Group of nodes [{}] will be pruned together.".format(", ".join(cluster_nodes_names)))

    def _is_module_prunable(self, graph: NNCFGraph, node: NNCFNode) -> Tuple[bool, str]:
        """
        Check whether we should prune module corresponding to provided node
        according to algorithm parameters.

        :param graph: Graph to work with.
        :param node: Node to check.
        :return: Tuple (prune, msg) where prune means whether we should/shouldn't prune module,
            msg is additional information why we should/shouldn't prune.
        """
        prune = True
        msg = None

        stop_propagation_ops = self._stop_propagation_op_metatype.get_all_op_aliases()
        types_to_track = self._prune_operations + stop_propagation_ops
        input_non_pruned_nodes = get_first_nodes_of_type(graph, types_to_track)
        output_non_pruned_nodes = get_last_nodes_of_type(graph, types_to_track)
        module_identifier = get_module_identifier(node)

        if not should_consider_scope(module_identifier, self._target_scopes, self._ignored_scopes):
            msg = 'Ignored adding Weight Pruner in: {}'.format(module_identifier)
            prune = False
        elif not self._prune_first and node in input_non_pruned_nodes:
            msg = 'Ignored adding Weight Pruner in: {} because'\
                             ' this scope is one of the first convolutions'.format(module_identifier)
            prune = False
        elif not self._prune_last and node in output_non_pruned_nodes:
            msg = "Ignored adding Weight Pruner in: {} because"\
                             " this scope is one of the last convolutions".format(module_identifier)
            prune = False
        elif is_grouped_conv(node) and not is_depthwise_conv(node):
            msg = 'Ignored adding Weight Pruner in: {} because' \
                  ' this scope is grouped convolution'.format(module_identifier)
            prune = False
        elif not self._prune_downsample_convs and is_conv_with_downsampling(node):
            msg = 'Ignored adding Weight Pruner in: {} because'\
                             ' this scope is convolution with downsample'.format(module_identifier)
            prune = False
        return prune, msg
