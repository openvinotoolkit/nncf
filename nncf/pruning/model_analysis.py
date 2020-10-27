from nncf.pruning.utils import _find_next_nodes_of_types

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

    def union_clusters(self, first_id, second_id):
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


def unit_groups_of_clusters(nodes_to_merge, clusterization):
    # TODO: refactor this function to something more clear
    if len(nodes_to_merge) <= 1:
        return

    max_order_node_id = max([(node.node_id, clusterization.get_cluster_for_node(node.node_id).importance) for node in nodes_to_merge], key=lambda x: x[1])[0]
    max_order_cluster_id = clusterization.get_cluster_for_node(max_order_node_id).id
    for node in nodes_to_merge:
        if node.node_id != max_order_node_id:
            current_node_cluster_id = clusterization.get_cluster_for_node(node.node_id).id
            clusterization.union_clusters(max_order_cluster_id, current_node_cluster_id)


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
    # TODO: rewrite (need new function to find next nodes and this list will not be needed anymore)
    other_types = set([graph.node_type_fn(nx_graph.nodes[node_name])for node_name in nx_graph.nodes if graph.node_type_fn(nx_graph.nodes[node_name]) not in identity_types])  # not identity
    topologically_sorted_nodes = [nx_graph.nodes[node_name] for node_name in nx.topological_sort(nx_graph)]

    all_special_nodes = [nx_graph.nodes[node_name] for node_name in nx_graph.nodes if graph.node_type_fn(nx_graph.nodes[node_name]) in special_types]

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

        # TODO: rewrite (need new function to find next nodes)
        all_outputs = _find_next_nodes_of_types(model, nncf_node, other_types)
        all_output_special_nodes = [node for node in all_outputs if node.op_exec_context.operator_name in special_types]
        if graph.node_type_fn(node) in special_types:
            all_output_special_nodes.append(nncf_node)
        unit_groups_of_clusters(all_output_special_nodes, clusterization)

    return clusterization