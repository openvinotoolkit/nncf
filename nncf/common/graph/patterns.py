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
from typing import Optional
from typing import List
from typing import Tuple
from typing import Hashable

import os
import copy
import itertools as it

import networkx as nx
import networkx.algorithms.isomorphism as ism


class QuantizationIgnorePatterns:
    def __init__(self):
        self._patterns_dict = dict()
        self.full_pattern_graph = GraphPattern()

    def register(self, pattern: 'GraphPattern', name: str, match: bool = True) -> None:
        if name in self._patterns_dict:
            raise KeyError('{} is already registered'.format(name))
        self._patterns_dict[name] = pattern
        if match:
            self.full_pattern_graph.add_pattern_alternative(pattern)

    def get_full_pattern_graph(self) -> 'GraphPattern':
        return self.full_pattern_graph

    def visualize_all_matching_pattern(self, path: str) -> None:
        self.full_pattern_graph.dump_graph(path)

    def visualize_all_patterns(self, dir_path: str) -> None:
        for patten_name, pattern in self._patterns_dict.items():
            pattern.dump_graph(os.path.join(dir_path, patten_name + '.dot'))

    def visualize_pattern(self, pattern_name: str, path: str) -> None:
        self._patterns_dict[pattern_name].dump_graph(os.path.join(path))


def _merge_two_patterns_alternative_to_this(first_pattern: 'GraphPattern', other_pattern: 'GraphPattern'):
    first_pattern_graph = first_pattern.graph
    other_graph = other_pattern.graph
    node_mapping = {}
    for node in other_graph.nodes:
        node_mapping[node] = first_pattern.node_counter
        first_pattern.node_counter += 1
    other_graph_copy = nx.relabel_nodes(other_graph, node_mapping, copy=True)
    return nx.union(first_pattern_graph, other_graph_copy)


def create_copy_of_subgraph(pattern: 'GraphPattern', subgraph: nx.DiGraph) -> nx.DiGraph:
    mapping = {}
    for node in subgraph.nodes:
        new_node = pattern.node_counter
        mapping[node] = new_node
        pattern.node_counter += 1
    return nx.relabel_nodes(subgraph, mapping, copy=True)


class GraphPattern:
    """
    Describes layer patterns in model's graph that should be considered as a single node
    during quantizer arrangement search algorithm

    :param _graph: Graph contains layer pattern/patterns
    """
    PATTERN_INPUT_NODE_TYPE = 'INPUT_NODE'

    def __init__(self):
        """
        :param types: List or single string of backend operations names
         that should be considered as one single node
        """
        self._graph = nx.DiGraph()
        self.node_counter = 0

    def __add__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add DiGraph nodes of other to self and add edge between
        last node of self's graph and first node of other's graph.

        The first and last nodes are found by nx.lexicographical_topological_sort().

        For more complex cases that are not covered by this function, use `join_patterns()`.

        :param other: GraphPattern that will be added
        """

        def _add_second_subgraph_to_first_with_connected_edge(first_graph: nx.DiGraph,
                                                              second_graph: nx.DiGraph) -> nx.DiGraph:

            union_graph = nx.union(first_graph, second_graph)

            first_graph_nodes = list(nx.lexicographical_topological_sort(first_graph, key=int))
            last_node_first_graph = first_graph_nodes[-1]
            assert first_graph.out_degree(last_node_first_graph) == 0
            second_nodes = list(nx.lexicographical_topological_sort(second_graph, key=int))
            first_node_second_graph = second_nodes[0]
            assert second_graph.in_degree(first_node_second_graph) == 0
            # Special case whtn first node is PATTERN_INPUT_NODE_TYPE
            if second_graph.nodes[first_node_second_graph]['type'][0] == GraphPattern.PATTERN_INPUT_NODE_TYPE:
                successors = union_graph.successors(first_node_second_graph)
                new_edges = list(it.product([last_node_first_graph], successors))
                union_graph.add_edges_from(new_edges)
                union_graph.remove_node(first_node_second_graph)
            else:
                union_graph.add_edge(last_node_first_graph, first_node_second_graph)
            return union_graph

        # final_pattern = GraphPattern(self.name + '+' + other.name)
        final_pattern = GraphPattern()
        weakly_self_components = self.get_weakly_connected_subgraphs()
        weakly_other_components = other.get_weakly_connected_subgraphs()
        for self_subgraph in weakly_self_components:
            for other_subgraph in weakly_other_components:
                # As this operation should output all graph combinations
                # It is essential to create copies of subgraphs and
                # add merge all possible connections
                # A: (a) (b)
                # B: (c) (d)
                #              (a)  (a_copy)  (b)    (b_copy)
                # A + B ---->   |       |      |        |
                #              (c)     (d)  (c_copy) (d_copy)
                #
                subgraph_copy = create_copy_of_subgraph(final_pattern, self_subgraph)
                other_subgraph_copy = create_copy_of_subgraph(final_pattern, other_subgraph)
                subgraph_copy = _add_second_subgraph_to_first_with_connected_edge(subgraph_copy, other_subgraph_copy)
                final_pattern.graph = nx.union(final_pattern.graph, subgraph_copy)

        return final_pattern

    def __or__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add other's DiGraph nodes to self's DiGraph as a new weakly connected components.
        It is a syntax sugar of 'add_pattern_alternative()'

        :param other: GraphPattern that will be added
        """
        new_pattern = copy.deepcopy(self)
        new_pattern.graph = _merge_two_patterns_alternative_to_this(new_pattern, other)
        return new_pattern

    def __eq__(self, other: 'GraphPattern') -> bool:
        return ism.is_isomorphic(self.graph, other.graph)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    @graph.setter
    def graph(self, graph: nx.DiGraph):
        self._graph = graph

    def add_pattern_alternative(self, other: 'GraphPattern') -> None:
        """
        Add 'other' pattern as a connected weakly component to 'self' pattern.

        :param other: GraphPattern that will be added
        """
        self.graph = _merge_two_patterns_alternative_to_this(self, other)

    def join_patterns(self, other: 'GraphPattern',
                      edges: Optional[List[Tuple[Hashable, Hashable]]] = None) -> None:
        """
        Add 'other' pattern to 'self' pattern and connect nodes from self to other determined by 'edges'.
        If edges is None, add edge between
        last node of self's graph and first node of other's graph.

        The first and last nodes are found by nx.lexicographical_topological_sort().

        :param other: GraphPattern that will be added
        :param edges: List of edges between self and other graphs.
            Edges must begin at self and finish at other.
        """
        self_graph = self.graph
        other_graph = other.graph
        node_mapping = {}
        for node in other_graph.nodes:
            node_mapping[node] = self.node_counter
            self.node_counter += 1
        other_graph_copy = nx.relabel_nodes(other.graph, node_mapping, copy=True)

        if edges is not None:
            new_edges = []
            for edge in edges:
                new_edge = (edge[0], node_mapping[edge[1]])
                new_edges.append(new_edge)

        union_graph = nx.union(self.graph, other_graph_copy)

        if edges is None:
            last_node_self = list(nx.lexicographical_topological_sort(self_graph, key=int))[-1]
            assert self_graph.out_degree(last_node_self) == 0
            first_node_other = list(nx.lexicographical_topological_sort(other_graph_copy, key=int))[0]
            assert other_graph_copy.in_degree(first_node_other) == 0
            # Special case with first node is PATTERN_INPUT_NODE_TYPE
            if other_graph_copy.nodes[first_node_other]['type'][0] == GraphPattern.PATTERN_INPUT_NODE_TYPE:
                successors = union_graph.successors(first_node_other)
                new_edges = list(it.product([last_node_self], successors))
                union_graph.add_edges_from(new_edges)
                union_graph.remove_node(first_node_other)
            else:
                union_graph.add_edge(last_node_self, first_node_other)
        else:
            union_graph.add_edges_from(new_edges)
        self._graph = union_graph

    def add_node(self, **attrs) -> int:
        if 'type' in attrs:
            if not isinstance(attrs['type'], list):
                attrs['type'] = [attrs['type']]
        self.graph.add_node(self.node_counter, **attrs)
        self.node_counter += 1
        return self.node_counter - 1

    def add_edge(self, u_name, v_name) -> None:
        self.graph.add_edge(u_name, v_name)

    def get_weakly_connected_subgraphs(self) -> List[nx.DiGraph]:
        return [self.graph.subgraph(c) for c in nx.weakly_connected_components(self.graph)]

    def dump_graph(self, path: str) -> None:
        nx.drawing.nx_pydot.write_dot(self.graph, path)
