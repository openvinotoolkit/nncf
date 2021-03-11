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

from typing import List

import networkx as nx

from nncf.dynamic_graph.graph import NNCFNode, NNCFGraph
from nncf.pruning.export_helpers import DefaultMetaOp
from nncf.pruning.utils import is_depthwise_conv, find_next_nodes_not_of_types
from nncf.pruning.export_utils import PruningOperationsMetatypeRegistry


# pylint: disable=protected-access
class NodesCluster:
    def __init__(self, cluster_id: int, nodes: List, nodes_orders: List[int]):
        self.id = cluster_id
        self.nodes = list(nodes)
        self.importance = max(nodes_orders)

    def clean_cluster(self):
        self.nodes = []
        self.importance = 0

    def add_nodes(self, nodes: List, importance: int):
        self.nodes.extend(nodes)
        self.importance = max(self.importance, importance)


class Clusterization:
    def __init__(self, id_attr_name="id"):
        self.clusters = {}
        self._node_to_cluster = {}
        self._id_attr = id_attr_name

    def get_cluster_by_id(self, cluster_id: int) -> NodesCluster:
        if cluster_id not in self.clusters:
            raise IndexError('No cluster with id = {}'.format(cluster_id))
        return self.clusters[cluster_id]

    def get_cluster_by_node_id(self, node_id: int) -> NodesCluster:
        if node_id not in self._node_to_cluster:
            raise IndexError('No cluster for node with id = {}'.format(node_id))
        return self.get_cluster_by_id(self._node_to_cluster[node_id])

    def is_node_in_clusterization(self, node_id: int) -> bool:
        return node_id in self._node_to_cluster

    def add_cluster(self, cluster: NodesCluster):
        cluster_id = cluster.id
        if cluster_id in self.clusters:
            raise IndexError('Cluster with index = {} already exist'.format(cluster_id))
        self.clusters[cluster_id] = cluster
        for node in cluster.nodes:
            self._node_to_cluster[getattr(node, self._id_attr)] = cluster_id

    def delete_cluster(self, cluster_id: int):
        if cluster_id not in self.clusters:
            raise IndexError('No cluster with index = {} to delete'.format(cluster_id))
        for node in self.clusters[cluster_id].nodes:
            node_id = getattr(node, self._id_attr)
            self._node_to_cluster.pop(node_id)
        self.clusters.pop(cluster_id)

    def get_all_clusters(self) -> List[NodesCluster]:
        return list(self.clusters.values())

    def get_all_nodes(self) -> List:
        all_nodes = []
        for cluster in self.clusters.values():
            all_nodes.extend(cluster.nodes)
        return all_nodes

    def merge_clusters(self, first_id: int, second_id: int):
        cluster_1 = self.get_cluster_by_id(first_id)
        cluster_2 = self.get_cluster_by_id(second_id)
        if cluster_1.importance > cluster_2.importance:
            cluster_1.add_nodes(cluster_2.nodes, cluster_2.importance)
            for node in cluster_2.nodes:
                self._node_to_cluster[getattr(node, self._id_attr)] = first_id
            self.clusters.pop(second_id)
        else:
            cluster_2.add_nodes(cluster_1.nodes, cluster_1.importance)
            for node in cluster_1.nodes:
                self._node_to_cluster[getattr(node, self._id_attr)] = second_id
            self.clusters.pop(first_id)

    def merge_list_of_clusters(self, clusters: List[int]):
        clusters = list(set(clusters))
        clusters.sort(key=lambda cluster_id: self.get_cluster_by_id(cluster_id).importance)
        for cluster_id in clusters[:-1]:
            self.merge_clusters(clusters[-1], cluster_id)


def get_position(nx_nodes_list, idx):
    for i, node in enumerate(nx_nodes_list):
        if node['id'] == idx:
            return i
    return None


def merge_clusters_for_nodes(nodes_to_merge: List[NNCFNode], clusterization: Clusterization):
    """
    Merges clusters to which nodes from nodes_to_merge belongs.
    :param nodes_to_merge: all nodes are clusters for which should be тerged
    :param clusterization:
    """
    if len(nodes_to_merge) <= 1:
        return

    # Will merge cluster with highest importance with others pairwise
    max_importance_node_id = None
    max_importance = 0
    for node in nodes_to_merge:
        importance = clusterization.get_cluster_by_node_id(node.node_id).importance
        if importance > max_importance:
            max_importance_node_id = node.node_id
            max_importance = importance

    max_importance_cluster_id = clusterization.get_cluster_by_node_id(max_importance_node_id).id
    for node in nodes_to_merge:
        if node.node_id != max_importance_node_id:
            current_node_cluster_id = clusterization.get_cluster_by_node_id(node.node_id).id
            clusterization.merge_clusters(max_importance_cluster_id, current_node_cluster_id)


def cluster_special_ops(graph: NNCFGraph, special_types: List[str], identity_types: List[str]) -> Clusterization:
    """
    This model will cluster all operations with type from special_types. Connected nodes is nodes that:
        1. Have path between nodes with only identity type nodes on it
        2. Have common input (identity type nodes can be on path from this input)
    :param graph: graph to work with
    :param special_types: list of types that should be grouped to groups of dependent nodes
    :return: Clusterization of special_types nodes to the dependent groups
    """
    nx_graph = graph._nx_graph
    topologically_sorted_nodes = [nx_graph.nodes[node_name] for node_name in nx.topological_sort(nx_graph)]
    all_special_nodes = [nx_graph.nodes[node_name] for node_name in nx_graph.nodes
                         if graph.node_type_fn(nx_graph.nodes[node_name]) in special_types]

    # 0. Initially all nodes is a separate clusters
    clusterization = Clusterization("node_id")
    for i, node in enumerate(all_special_nodes):
        nncf_node = graph._nx_node_to_nncf_node(node)
        cluster = NodesCluster(i, [nncf_node], [get_position(topologically_sorted_nodes, node['id'])])
        clusterization.add_cluster(cluster)

    for node in topologically_sorted_nodes:
        if graph.node_type_fn(node) in identity_types:
            continue

        nncf_node = graph._nx_node_to_nncf_node(node)

        all_outputs = find_next_nodes_not_of_types(graph, nncf_node, identity_types)
        all_output_special_nodes = [node for node in all_outputs
                                    if node.op_exec_context.operator_name in special_types]
        if graph.node_type_fn(node) in special_types:
            all_output_special_nodes.append(nncf_node)
        merge_clusters_for_nodes(all_output_special_nodes, clusterization)

    return clusterization


class ModelAnalyzer:
    """
    Analyze the model before pruning to understand which parts could potentially be pruned without conflicts
     (all nodes that can't get pruned input will receive a non-pruned input).

    The algorithm consists of three steps:
    1. Set attribute accept_pruned_input to all nodes. This attribute shows can this node potentially get
     pruned input or node.
    2.  Calculate can_prune attribute for all nodes by propagating accept_pruned_input up
     (from the result of the network to the inputs). Node can be pruned if all outputs of this node accept
      pruned input and all outputs can be pruned.
    3. Propagates can_prune down from input nodes to the outputs.

    As a result, all nodes marked by the can_prune attribute as potentially prunable or not.
    """
    def __init__(self, graph: NNCFGraph,
                 pruning_operator_metatypes: PruningOperationsMetatypeRegistry):
        self.graph = graph
        self._nx_graph = self.graph._nx_graph

        self._pruning_operator_metatypes = pruning_operator_metatypes
        pruning_op_metatypes_dict = self._pruning_operator_metatypes.registry_dict
        self._stop_propagation_op_metatype = pruning_op_metatypes_dict['stop_propagation_ops']
        self._concat_op_metatype = pruning_op_metatypes_dict['concat']
        self._convolution_op_metatype = pruning_op_metatypes_dict['convolution']

        self.can_prune = {idx: True for idx in self.graph.get_all_node_idxs()}
        self.accept_pruned_input = {idx: True for idx in self.graph.get_all_node_idxs()}

    def node_propagate_can_prune_attr(self, nncf_node: NNCFNode) -> bool:
        """
        Whether node propagates can_prune attr through. That means a node can propagate pruning mask
         (for example,  activations propagate mask, but convolutions stop mask propagation)
        :param nncf_node: node to work with
        :return: bool: propagates this node can_prune throw or not
        """
        node_type = nncf_node.op_exec_context.operator_name
        is_conv = node_type in self._convolution_op_metatype.get_all_op_aliases()
        return not is_conv or (is_conv and is_depthwise_conv(nncf_node))

    def node_accept_different_inputs(self, nncf_node: NNCFNode) -> bool:
        """
        Return whether nx_node accept pruned and not pruned inputs as inputs at the same time.
        """
        node_type = nncf_node.op_exec_context.operator_name
        return node_type in self._concat_op_metatype.get_all_op_aliases()

    def get_class_by_type_name(self, type_name: str) -> DefaultMetaOp:
        """
        Return class of metaop that corresponds to type_name type.
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
        reversed_sorted_nodes = reversed([self._nx_graph.nodes[name] for name in nx.topological_sort(self._nx_graph)])
        for nx_node in reversed_sorted_nodes:
            nncf_node = self.graph._nx_node_to_nncf_node(nx_node)

            # Check all output nodes accept_pruned_input attribute
            out_nodes = self.graph.get_next_nodes(nncf_node)
            outputs_accept_pruned_input = all(self.accept_pruned_input[node.node_id] for node in out_nodes)

            # Check all output nodes can_prune attribute
            outputs_will_be_pruned = all([self.can_prune[node.node_id]
                                          for node in out_nodes if self.node_propagate_can_prune_attr(node)])
            self.can_prune[nncf_node.node_id] = outputs_accept_pruned_input and outputs_will_be_pruned

    def propagate_can_prune_attr_down(self):
        """
        Propagating can_prune attribute down to fix all branching cases with one pruned and one not pruned
        branches.
        """
        sorted_nodes = [self._nx_graph.nodes[name] for name in nx.topological_sort(self._nx_graph)]
        for nx_node in sorted_nodes:
            nncf_node = self.graph._nx_node_to_nncf_node(nx_node)
            # Propagate attribute only in not conv case
            if self.node_propagate_can_prune_attr(nncf_node):
                in_nodes = self.graph.get_previous_nodes(nncf_node)
                can_prune = all([self.can_prune[node.node_id] for node in in_nodes])
                can_prune_any = any([self.can_prune[node.node_id] for node in in_nodes])

                if (not self.node_accept_different_inputs(nncf_node) and not can_prune) or \
                        (self.node_accept_different_inputs(nncf_node) and not can_prune_any):
                    self.can_prune[nncf_node.node_id] = can_prune

    def set_accept_pruned_input_attr(self):
        for nncf_node in self.graph.get_all_nodes():
            node_type = nncf_node.op_exec_context.operator_name
            cls = self.get_class_by_type_name(node_type)
            self.accept_pruned_input[nncf_node.node_id] = cls.accept_pruned_input(nncf_node)

    def analyse_model_before_pruning(self):
        self.set_accept_pruned_input_attr()
        self.propagate_can_prune_attr_up()
        self.propagate_can_prune_attr_down()
        return self.can_prune
