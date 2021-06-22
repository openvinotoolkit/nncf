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
from typing import Union
from typing import List
from typing import Tuple
from typing import Hashable

import copy

import networkx as nx
import networkx.algorithms.isomorphism as ism


def create_copy_of_subgraph(subgraph: nx.DiGraph) -> nx.DiGraph:
    mapping = {}
    for node in subgraph.nodes:
        new_node = GraphPattern.NODE_COUNTER
        mapping[node] = new_node
        GraphPattern.NODE_COUNTER += 1
    return nx.relabel_nodes(subgraph, mapping, copy=True)


def add_subgraph_to_graph(graph: nx.DiGraph, subgraph: nx.DiGraph) -> None:
    for node in subgraph.nodes:
        if node in graph.nodes:
            assert False
        graph.add_node(node, type=subgraph.nodes[node]['type'])
    for edge in subgraph.edges:
        graph.add_edge(edge[0], edge[1])


class GraphPattern:
    """
    Describes layer patterns in model's graph that should be considered as a single node
    during quantizer arrangement search algorithm

    :param _graph: Graph contains layer pattern/patterns
    """
    NODE_COUNTER = 0

    def __init__(self, types: Optional[Union[str, List[str]]] = None):
        """
        :param types: List or signle string of backend operations names
         that should be considered as one single node
        """
        self._graph = nx.DiGraph()
        if types is not None:
            if isinstance(types, str):
                types = [types]
            self._graph.add_node(GraphPattern.NODE_COUNTER, type=types)
            GraphPattern.NODE_COUNTER += 1

    def __add__(self, other):
        """
        Add DiGraph nodes of other to self and add edge between
        last node of self's graph and first node of other's graph

        The first and last nodes are found by nx.topological_sort()

        Please, use this operand when the resulted pattern is quite
        straightforward.

        If it is not your case use fuse_patterns() instead
        """

        def _add_second_subgraph_to_first_with_connected_edge(first_graph: nx.DiGraph,
                                                              second_graph: nx.DiGraph) -> nx.DiGraph:
            union_graph = nx.union(first_graph, second_graph)

            first_graph_nodes = list(nx.topological_sort(first_graph))
            last_node_first_graph = first_graph_nodes[-1]
            assert first_graph.out_degree(last_node_first_graph) == 0
            second_nodes = list(nx.topological_sort(second_graph))
            first_node_second_graph = second_nodes[0]
            assert second_graph.in_degree(first_node_second_graph) == 0

            union_graph.add_edge(last_node_first_graph, first_node_second_graph)
            return union_graph

        final_graph = nx.DiGraph()
        weakly_self_subgraphs = self.get_weakly_connected_subgraphs()
        weakly_other_subgraphs = other.get_weakly_connected_subgraphs()
        for self_subgraph in weakly_self_subgraphs:
            for other_subgraph in weakly_other_subgraphs:
                # As this operation should output all graph combinations
                # It is essential to create copies of subgraphs and
                # add merge all possible connections
                # A: (a) (b)
                # B: (c) (d)
                #              (a)  (a_copy)  (b)    (b_copy)
                # A + B ---->   |       |      |        |
                #              (c)     (d)  (c_copy) (d_copy)
                #
                subgraph_copy = create_copy_of_subgraph(self_subgraph)
                other_subgraph_copy = create_copy_of_subgraph(other_subgraph)
                subgraph_copy = _add_second_subgraph_to_first_with_connected_edge(subgraph_copy, other_subgraph_copy)
                add_subgraph_to_graph(final_graph, subgraph_copy)

        final_pattern = copy.deepcopy(self)
        final_pattern._graph = final_graph
        return final_pattern

    def __or__(self, other):
        """
        Add other's DiGraph nodes to self's DiGraph.
        Other's DiGraph will be placed as a new weakly connected components
        """
        new_pattern = copy.deepcopy(self)
        other_copy = create_copy_of_subgraph(other.graph)
        add_subgraph_to_graph(new_pattern._graph, other_copy)
        return new_pattern

    def __eq__(self, other):
        return ism.is_isomorphic(self.graph, other.graph)

    @property
    def graph(self):
        return self._graph

    def add_pattern_alternative(self, other: 'GraphPattern') -> None:
        other_graph = other.graph
        self_graph = self.graph
        for node in other_graph.nodes:
            if node in self_graph.nodes:
                assert False
            self_graph.add_node(node, type=other_graph.nodes[node]['type'])
        for edge in other_graph.edges:
            self_graph.add_edge(edge[0], edge[1])

    def join_patterns(self, other: 'GraphPattern', edges: Optional[List[Tuple[Hashable, Hashable]]] = None) -> None:
        if edges is None:
            first_node_other = list(nx.topological_sort(other.graph))[0]
            assert other.graph.in_degree(first_node_other) == 0
            last_node_self = list(nx.topological_sort(self.graph))[-1]
            assert self.graph.out_degree(last_node_self) == 0
            edges = [[last_node_self, first_node_other]]
        for edge in edges:
            assert edge[0] in self.graph.nodes
            assert edge[1] in other.graph.nodes
        add_subgraph_to_graph(self.graph, other.graph)
        for edge in edges:
            self.graph.add_edge(edge[0], edge[1])

    def add_node(self, t: List[str]) -> int:
        self.graph.add_node(GraphPattern.NODE_COUNTER, type=t)
        GraphPattern.NODE_COUNTER += 1
        return GraphPattern.NODE_COUNTER - 1

    def add_edge(self, u_name, v_name) -> None:
        self.graph.add_edge(u_name, v_name)

    def get_weakly_connected_subgraphs(self) -> List[nx.DiGraph]:
        return [self.graph.subgraph(c) for c in nx.weakly_connected_components(self.graph)]

    def dump_graph(self, path: str) -> None:
        nx.drawing.nx_pydot.write_dot(self.graph, path)
