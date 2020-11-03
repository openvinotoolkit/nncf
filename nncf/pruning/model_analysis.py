from nncf.pruning.export_helpers import PRUNING_OPERATOR_METATYPES, Convolution, \
    Concat, StopMaskForwardOps

from nncf.pruning.utils import is_depthwise_conv, find_next_nodes_not_types, find_next_nodes_of_types

from nncf.nncf_network import NNCFNetwork
import networkx as nx


class NodesCluster:
    def __init__(self, id, nodes, nodes_orders):
        self.id = id
        self.nodes = nodes
        self.importance = max(nodes_orders)

    def clean_cluster(self):
        self.nodes = []
        self.importance = 0

    def add_nodes(self, nodes):
        self.nodes.extend(nodes)


class Clusterization:
    def __init__(self, id_attr_name="id"):
        self.clusters = {}
        self._node_to_cluster = {}
        self.id_attr = id_attr_name

    def get_cluster_by_id(self, id):
        if id in self.clusters:
            return self.clusters[id]
        else:
            raise IndexError('No cluster with such index')

    def get_cluster_for_node(self, node_id):
        return self.clusters[self._node_to_cluster[node_id]]

    def is_not_in_any_cluster(self, node_id):
        return node_id in self._node_to_cluster

    def add_cluster(self, cluster: NodesCluster, id):
        if id in self.clusters:
            raise IndexError
        else:
            self.clusters[id] = cluster
            for node in cluster.nodes:
                self._node_to_cluster[getattr(node, self.id_attr)] = id

    def delete_cluster(self, id):
        for node in self.clusters[id].nodes:
            node_id = getattr(node, self.id_attr)
            self._node_to_cluster.pop(node_id)
        self.clusters.pop(id)

    def get_all_clusters(self):
        return self.clusters.values()

    def get_all_nodes(self):
        all_nodes = []
        for cluster in self.clusters.values():
            all_nodes.extend(cluster.nodes)
        return all_nodes

    def merge_clusters(self, first_id, second_id):
        cluster_1 = self.clusters[first_id]
        cluster_2 = self.clusters[second_id]
        if cluster_1.importance > cluster_2.importance:
            cluster_1.add_nodes(cluster_2.nodes)
            for node in cluster_2.nodes:
                self._node_to_cluster[getattr(node, self.id_attr)] = first_id
            self.clusters.pop(second_id)
        else:
            cluster_2.add_nodes(cluster_1.nodes)
            for node in cluster_1.nodes:
                self._node_to_cluster[getattr(node, self.id_attr)] = second_id
            self.clusters.pop(first_id)


def get_position(nx_nodes_list, idx):
    for i, node in enumerate(nx_nodes_list):
        if node['id'] == idx:
            return i


def unit_clusters_for_nodes(nodes_to_merge, clusterization):
    """

    :param nodes_to_merge: all nodes are clusters for which should be тerged
    :param clusterization:
    """
    if len(nodes_to_merge) <= 1:
        return

    # Will merge cluster with highest importance with others pairwise
    max_importance_node_id = None
    max_importance = 0
    for node in nodes_to_merge:
        importance = clusterization.get_cluster_for_node(node.node_id).importance
        if importance > max_importance:
            max_importance_node_id = node.node_id
            max_importance = importance

    max_importance_cluster_id = clusterization.get_cluster_for_node(max_importance_node_id).id
    for node in nodes_to_merge:
        if node.node_id != max_importance_node_id:
            current_node_cluster_id = clusterization.get_cluster_for_node(node.node_id).id
            clusterization.merge_clusters(max_importance_cluster_id, current_node_cluster_id)


def cluster_special_ops_in_model(model: object, special_types: object, identity_types: object) -> Clusterization:
    """
    This model will cluster all operations with type from special_types. Connected nodes is nodes that:
        1. Have path between nodes with only identity type nodes on it
        2. Have common input (identity type nodes can be on path from this input)
    :param model:
    :param special_types:
    :return: list of lists with clusters of special_types nodes (of type NNCFNode)
    """
    graph = model.get_original_graph()
    nx_graph = graph._nx_graph
    topologically_sorted_nodes = [nx_graph.nodes[node_name] for node_name in nx.topological_sort(nx_graph)]
    all_special_nodes = [nx_graph.nodes[node_name] for node_name in nx_graph.nodes
                         if graph.node_type_fn(nx_graph.nodes[node_name]) in special_types]

    # 0. Initially all nodes is a separate clusters
    clusterization = Clusterization("node_id")
    for i, node in enumerate(all_special_nodes):
        nncf_node = graph._nx_node_to_nncf_node(node)
        cluster = NodesCluster(i, [nncf_node], [get_position(topologically_sorted_nodes, node['id'])])
        clusterization.add_cluster(cluster, i)

    for node in topologically_sorted_nodes:
        if graph.node_type_fn(node) in identity_types:
            continue

        nncf_node = graph._nx_node_to_nncf_node(node)

        all_outputs = find_next_nodes_not_types(model, nncf_node, identity_types)
        all_output_special_nodes = [node for node in all_outputs if node.op_exec_context.operator_name in special_types]
        if graph.node_type_fn(node) in special_types:
            all_output_special_nodes.append(nncf_node)
        unit_clusters_for_nodes(all_output_special_nodes, clusterization)

    return clusterization


class ModelAnalyser:
    """
    Analyse model architecture before pruning
    """
    def __init__(self, target_model: NNCFNetwork):
        self.model = target_model
        self.graph = target_model.get_original_graph()
        self.nx_graph = self.graph._nx_graph

        self.can_prune = {idx: True for idx in self.graph.get_all_node_idxs()}
        self.accept_pruned_input = {idx: True for idx in self.graph.get_all_node_idxs()}

    def node_propagate_can_prune_attr(self, nncf_node):
        node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
        node_type = nncf_node.op_exec_context.operator_name
        is_conv = node_type in Convolution().get_all_op_aliases()
        return not is_conv or (is_conv and is_depthwise_conv(node_module))

    def node_accept_different_inputs(self, nncf_node):
        """
        Return whether nx_node accept pruned and not pruned inputs as inputs at the same time.
        """
        node_type = nncf_node.op_exec_context.operator_name
        return node_type in Concat.get_all_op_aliases()

    @staticmethod
    def get_class_by_type_name(type_name):
        """
        Return class of metaop that corresponds to type_name type.
        """
        cls = PRUNING_OPERATOR_METATYPES.get_operator_metatype_by_op_name(type_name)
        if cls is None:
            cls = StopMaskForwardOps
        return cls

    def propagate_can_prune_attr_up(self):
        """
        Propagating can_prune attribute in reversed topological order.
        This attribute depends on accept_pruned_input and can_prune attributes of output nodes.
        Node can_prune is True if all outputs accept_pruned_input is True and all outputs
        (except convs because conv can be pruned by input and output independently).
        """
        reversed_sorted_nodes = reversed([self.nx_graph.nodes[name] for name in nx.topological_sort(self.nx_graph)])
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
        sorted_nodes = [self.nx_graph.nodes[name] for name in nx.topological_sort(self.nx_graph)]
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
            node_module = self.model.get_module_by_scope(nncf_node.op_exec_context.scope_in_model)
            node_type = nncf_node.op_exec_context.operator_name
            cls = self.get_class_by_type_name(node_type)()
            self.accept_pruned_input[nncf_node.node_id] = cls.accept_pruned_input(self.model, self.graph, node_module)

    def analyse_model_before_pruning(self):
        self.set_accept_pruned_input_attr()
        self.propagate_can_prune_attr_up()
        self.propagate_can_prune_attr_down()
        return self.can_prune
