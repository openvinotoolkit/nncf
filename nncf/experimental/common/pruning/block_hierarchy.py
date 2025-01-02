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

from pathlib import Path
from typing import Dict, List

import networkx as nx

from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.experimental.common.pruning.propagation_data import PropagationGroup


class BlockHierarchy:
    """
    Graph that represents the hierarchy of propagation blocks/groups
    """

    PROPAGATION_GROUP = "propagation_group"

    def __init__(self, root_groups: List[PropagationGroup]) -> None:
        """
        Creates hierarchy of propagation blocks/groups by traversing children in the given root groups.

        :param roots: list of the root groups
        :return: networkx graph that represents the hierarchy of propagation blocks/groups.
        """

        self._id_counter = 0
        self._graph = nx.DiGraph()
        self._visited_block_ids_map: Dict[int, int] = {}

        for root_group in root_groups:
            self._add_group_to_graph(root_group)
            self._id_counter += 1

    def get_groups_on_leaves(self) -> List[PropagationGroup]:
        """
        Returns the list of all propagation groups on the leaves.
        """
        groups = []
        for node_id, data in self._graph.nodes(data=True):
            is_leaf = self._graph.out_degree(node_id) == 0
            if is_leaf:
                groups.append(data[self.PROPAGATION_GROUP])
        return groups

    def visualize_graph(self, path: Path) -> None:
        out_graph = self._get_graph_for_visualization()
        write_dot_graph(out_graph, path)

    def _add_group_to_graph(self, parent_group: PropagationGroup) -> None:
        """
        Recursive helper to traverse children of the given PropagationBlock with adding them to the graph.

        :param parent_group: current group for traversing.
        """
        parent_graph_id = str(self._id_counter)
        is_leaf = not parent_group.has_children()
        attrs = {self.PROPAGATION_GROUP: parent_group}
        self._graph.add_node(parent_graph_id, **attrs)
        self._visited_block_ids_map[id(parent_group)] = parent_graph_id
        if not is_leaf:
            for child_group in parent_group.get_children():
                child_id = id(child_group)
                if child_id not in self._visited_block_ids_map:
                    self._id_counter += 1
                    child_graph_id = str(self._id_counter)
                    self._visited_block_ids_map[id(child_group)] = child_graph_id
                    self._graph.add_edge(parent_graph_id, child_graph_id)
                    self._add_group_to_graph(child_group)
                else:
                    child_graph_id = self._visited_block_ids_map[child_id]
                    self._graph.add_edge(parent_graph_id, child_graph_id)

    def _get_graph_for_visualization(self) -> nx.DiGraph:
        """
        :return: A user-friendly graph .dot file, making it easier to debug the block hierarchy.
        """
        out_graph = nx.DiGraph()
        for node_id, node_data in self._graph.nodes(data=True):
            group = node_data[self.PROPAGATION_GROUP]
            is_leaf = not group.has_children()
            color = "grey"
            if is_leaf:
                color = "red" if group.is_invalid else "green"
            out_graph.add_node(node_id, label=f'"{str(group)}"', color=color, style="filled")

        for u, v in self._graph.edges:
            out_graph.add_edge(u, v)

        return out_graph
