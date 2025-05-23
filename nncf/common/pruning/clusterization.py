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

from typing import Callable, Generic, Hashable, Optional, TypeVar

T = TypeVar("T")


class Cluster(Generic[T]):
    """
    Represents element of Clusterization. Groups together elements.
    """

    def __init__(self, cluster_id: int, elements: list[T], nodes_orders: list[int]) -> None:
        self.id = cluster_id
        self.elements = list(elements)
        self.importance = max(nodes_orders)

    def clean_cluster(self) -> None:
        self.elements = []
        self.importance = 0

    def add_elements(self, elements: list[T], importance: int) -> None:
        self.elements.extend(elements)
        self.importance = max(self.importance, importance)


class Clusterization(Generic[T]):
    """
    Handles group of node clusters allowing to add new cluster,
    delete existing one or merge existing clusters.
    """

    def __init__(self, id_fn: Optional[Callable[[T], Hashable]] = None) -> None:
        self.clusters: dict[int, Cluster[T]] = {}
        self._element_to_cluster: dict[Hashable, int] = {}
        if id_fn is None:
            self._id_fn = lambda x: x.id
        else:
            self._id_fn = id_fn

    def get_cluster_by_id(self, cluster_id: int) -> Cluster[T]:
        """
        Returns cluster according to provided cluster_id.

        :param cluster_id: Id of the cluster.
        :return: Cluster according to provided `cluster_id`.
        """
        if cluster_id not in self.clusters:
            msg = f"No cluster with id = {cluster_id}"
            raise IndexError(msg)
        return self.clusters[cluster_id]

    def get_cluster_containing_element(self, element_id: Hashable) -> Cluster[T]:
        """
        Returns cluster containing element with provided `element_id`.

        :param element_id: Id of the element which is in cluster.
        :return: Cluster containing element with provided `element_id`.
        """
        if element_id not in self._element_to_cluster:
            msg = f"No cluster for node with id = {element_id}"
            raise IndexError(msg)
        return self.get_cluster_by_id(self._element_to_cluster[element_id])

    def is_node_in_clusterization(self, node_id: int) -> bool:
        """
        Returns whether node with provided `node_id` is in clusterization.

        :param node_id: Id of the node to test.
        :return: Whether node with provided `node_id` is in clusterization.
        """
        return node_id in self._element_to_cluster

    def add_cluster(self, cluster: Cluster[T]) -> None:
        """
        Adds provided cluster to clusterization.

        :param cluster: Cluster to add.
        """
        cluster_id = cluster.id
        if cluster_id in self.clusters:
            msg = f"Cluster with index = {cluster_id} already exist"
            raise IndexError(msg)
        self.clusters[cluster_id] = cluster
        for elt in cluster.elements:
            self._element_to_cluster[self._id_fn(elt)] = cluster_id  # type: ignore[no-untyped-call]

    def delete_cluster(self, cluster_id: int) -> None:
        """
        Removes cluster with `cluster_id` from clusterization.

        :param cluster_id: Id of a cluster to delete.
        """
        if cluster_id not in self.clusters:
            msg = f"No cluster with index = {cluster_id} to delete"
            raise IndexError(msg)
        for elt in self.clusters[cluster_id].elements:
            node_id = self._id_fn(elt)  # type: ignore[no-untyped-call]
            self._element_to_cluster.pop(node_id)
        self.clusters.pop(cluster_id)

    def get_all_clusters(self) -> list[Cluster[T]]:
        """
        Returns list of all clusters in clusterization.

        :return: List of all clusters in clusterization.
        """
        return list(self.clusters.values())

    def get_all_nodes(self) -> list[T]:
        """
        Returns list all elements of all clusters in clusterization.

        :return: List all elements of all clusters in clusterization.
        """
        all_elements = []
        for cluster in self.clusters.values():
            all_elements.extend(cluster.elements)
        return all_elements

    def merge_clusters(self, first_id: int, second_id: int) -> None:
        """
        Merges two clusters with provided ids.

        :param first_id: Id of the first cluster to merge.
        :param second_id: Id of the second cluster to merge.
        """
        cluster_1 = self.get_cluster_by_id(first_id)
        cluster_2 = self.get_cluster_by_id(second_id)
        if cluster_1.importance > cluster_2.importance:
            cluster_1.add_elements(cluster_2.elements, cluster_2.importance)
            for elt in cluster_2.elements:
                self._element_to_cluster[self._id_fn(elt)] = first_id  # type: ignore[no-untyped-call]
            self.clusters.pop(second_id)
        else:
            cluster_2.add_elements(cluster_1.elements, cluster_1.importance)
            for elt in cluster_1.elements:
                self._element_to_cluster[self._id_fn(elt)] = second_id  # type: ignore[no-untyped-call]
            self.clusters.pop(first_id)

    def merge_list_of_clusters(self, clusters: list[int]) -> None:
        """
        Merges provided clusters.

        :param clusters: List of clusters to merge.
        """
        clusters = list(set(clusters))
        clusters.sort(key=lambda x: self.get_cluster_by_id(x).importance)
        for cluster_id in clusters[:-1]:
            self.merge_clusters(clusters[-1], cluster_id)
