"""
 Copyright (c) 2019-2021 Intel Corporation
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
from typing import Optional
from typing import List
from typing import Tuple
from typing import Hashable

import os
import copy
import itertools as it

import networkx as nx
import networkx.algorithms.isomorphism as ism


class HWFusedPatterns:
    """
    Stores all layer patterns to be fused determined by hardware specific.
    This essence is used in the quantization algorithm.
    The operations in these patterns should be considered as a single
    during the quantization algorithm.
    """

    def __init__(self):
        self._patterns_dict = {}
        self._full_pattern_graph = GraphPattern()

    def register(self, pattern: 'GraphPattern', name: str, match: bool = True) -> None:
        """
        Registers new pattern.

        :param pattern: pattern to be added
        :param name: name associated with the pattern
        :param match: whether should the pattern used as fussing pattern
        """
        if name in self._patterns_dict:
            raise KeyError('{} is already registered'.format(name))
        self._patterns_dict[name] = pattern
        if match:
            self._full_pattern_graph.add_pattern_alternative(pattern)

    def get_full_pattern_graph(self) -> 'GraphPattern':
        return self._full_pattern_graph

    def visualize_full_pattern_graph(self, path: str) -> None:
        self._full_pattern_graph.dump_graph(path)

    def visualize_all_patterns(self, dir_path: str) -> None:
        """
        Dump graphs of all registered patterns to dir_path
        """
        for patten_name, pattern in self._patterns_dict.items():
            pattern.dump_graph(os.path.join(dir_path, patten_name + '.dot'))

    def visualize_pattern(self, pattern_name: str, path: str) -> None:
        self._patterns_dict[pattern_name].dump_graph(os.path.join(path))


class GraphPattern:
    """
    Describes layer patterns in model's graph.
    This class is used in quantizer arrangement search algorithm, representing layer fusing patterns

    :param ANY_PATTERN_NODE_TYPE: Special node type, meaning any type inside the pattern.
    :param NON_PATTERN_NODE_TYPE: Special node type, meaning any type outside the pattern.
    """
    ANY_PATTERN_NODE_TYPE = 'ANY_PATTERN_NODE'
    NON_PATTERN_NODE_TYPE = 'NON_PATTERN_NODE'

    def __init__(self):
        self._graph = nx.DiGraph()
        self._node_counter = 0

    def __add__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add DiGraph nodes of other to self and add edge between
        last node of self's graph and first node of other's graph.

        The first and last nodes are found by nx.lexicographical_topological_sort().

        For more complex cases that are not covered by this function, use `join_patterns()`.

        :param other: GraphPattern that will be added.
        :return: resulted GraphPattern.
        """

        final_pattern = GraphPattern()
        for self_subgraph in self.get_weakly_connected_subgraphs():
            for other_subgraph in other.get_weakly_connected_subgraphs():
                # As this operation should output all graph combinations
                # It is essential to create copies of subgraphs and
                # add merge all possible connections
                # A: (a) (b)
                # B: (c) (d)
                #              (a)  (a_copy)  (b)    (b_copy)
                # A + B ---->   |       |      |        |
                #              (c)     (d)  (c_copy) (d_copy)
                #
                subgraph_copy = final_pattern._unite_with_copy_of_graph(self_subgraph)
                other_subgraph_copy = final_pattern._unite_with_copy_of_graph(other_subgraph)
                final_pattern._add_edge_connected_subgraphs(subgraph_copy, other_subgraph_copy)

        return final_pattern

    def __or__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add other's DiGraph nodes to self's DiGraph as a new weakly connected components.
        It is a syntax sugar of 'add_pattern_alternative()'

        :param other: GraphPattern that will be added
        :return: resulted GraphPattern.
        """
        new_pattern = copy.deepcopy(self)
        new_pattern._unite_with_copy_of_graph(other.graph)
        return new_pattern

    def __eq__(self, other: 'GraphPattern') -> bool:
        return ism.is_isomorphic(self._graph, other.graph)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def _unite_with_copy_of_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        Creates a copy of 'graph', relabels node names according to self.node_counter
        and then unites relabeled graph with graph of 'self'.

        :param graph: graph, with which, self's graph will be united.
        :return: resulted graph.
        """
        mapping = {}
        for node in graph.nodes:
            new_node = self._node_counter
            mapping[node] = new_node
            self._node_counter += 1
        other_graph_copy = nx.relabel_nodes(graph, mapping, copy=True)
        self._graph = nx.union(self._graph, other_graph_copy)
        return other_graph_copy

    def _add_edge_connected_subgraphs(self,
                                      first_graph: nx.DiGraph,
                                      second_graph: nx.DiGraph) -> None:
        """
        Adds an edge between last node of 'first_graph' and first node of 'second_graph',
        which are found by nx.lexicographical_topological_sort().

        :param first_graph: the graph which will be traversed the first in the united graph.
        :param second_graph: the graph which will be traversed the second in the united graph.
        """
        self_graph = self._graph
        last_node_first_graph = list(nx.lexicographical_topological_sort(first_graph, key=int))[-1]
        assert first_graph.out_degree(last_node_first_graph) == 0
        first_node_second_graph = list(nx.lexicographical_topological_sort(second_graph, key=int))[0]
        assert second_graph.in_degree(first_node_second_graph) == 0

        # Special case when first node is ANY_PATTERN_NODE_TYPE or NON_PATTERN_NODE_TYPE
        if GraphPattern.ANY_PATTERN_NODE_TYPE in second_graph.nodes[first_node_second_graph]['type'] or \
                GraphPattern.NON_PATTERN_NODE_TYPE in second_graph.nodes[first_node_second_graph]['type']:
            successors = self_graph.successors(first_node_second_graph)
            new_edges = list(it.product([last_node_first_graph], successors))
            self_graph.add_edges_from(new_edges)
            self_graph.remove_node(first_node_second_graph)
        else:
            self_graph.add_edge(last_node_first_graph, first_node_second_graph)

    def add_pattern_alternative(self, other: 'GraphPattern') -> None:
        """
        Adds 'other' pattern as a weakly connected component to 'self' pattern.

        :param other: GraphPattern that will be added
        """
        self._unite_with_copy_of_graph(other.graph)

    def join_patterns(self, other: 'GraphPattern',
                      edges: Optional[List[Tuple[Hashable, Hashable]]] = None) -> None:
        """
        Adds 'other' pattern to 'self' pattern and connect nodes from self to other specified by 'edges'.

        If edges is None, adds an edge between
        last node of self's graph and first node of other's graph,
        which are found by nx.lexicographical_topological_sort().

        If other starts from a node with ANY_PATTERN_NODE_TYPE or NON_PATTERN_NODE_TYPE types,
        the input node of the other will be discarded from the final pattern.

        :param other: GraphPattern that will be added
        :param edges: List of edges between self and other graphs.
            Edges must begin at self and finish at other.
        """
        # Unite nodes
        other_graph_copy = copy.deepcopy(other.graph)
        node_mapping = {}
        for node_key in other_graph_copy.nodes:
            node_mapping[node_key] = self._node_counter
            self._node_counter += 1
        other_graph_copy = nx.relabel_nodes(other_graph_copy, node_mapping, copy=True)

        saved_graph = copy.deepcopy(self._graph)
        self._graph = nx.union(saved_graph, other_graph_copy)

        # Add edge/edges
        if edges is None:
            self._add_edge_connected_subgraphs(saved_graph, other_graph_copy)
        else:
            remapped_edges = []
            for edge in edges:
                new_edge = (edge[0], node_mapping[edge[1]])
                remapped_edges.append(new_edge)
            self._graph.add_edges_from(remapped_edges)

    def add_node(self, **attrs) -> int:
        if 'type' in attrs:
            if not isinstance(attrs['type'], list):
                attrs['type'] = [attrs['type']]
        self._graph.add_node(self._node_counter, **attrs)
        self._node_counter += 1
        return self._node_counter - 1

    def add_edge(self, u_name, v_name) -> None:
        self._graph.add_edge(u_name, v_name)

    def add_edges_from(self, ebunch_to_add, **attr) -> None:
        self._graph.add_edges_from(ebunch_to_add, **attr)

    def get_weakly_connected_subgraphs(self) -> List[nx.DiGraph]:
        return [self._graph.subgraph(c) for c in nx.weakly_connected_components(self._graph)]

    def dump_graph(self, path: str) -> None:
        nx.drawing.nx_pydot.write_dot(self._graph, path)


def merge_two_types_of_operations(first_op: Dict, second_op: Dict, label: str) -> Dict:
    res = {'type': first_op['type']}
    res['type'].extend(second_op['type'])
    res['label'] = label
    return res
