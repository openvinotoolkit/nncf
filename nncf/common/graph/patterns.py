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
import itertools as it

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
        graph.add_node(node, type=subgraph.nodes[node]['type'], label=subgraph.nodes[node]['label'])
    for edge in subgraph.edges:
        graph.add_edge(edge[0], edge[1])


class GraphPattern:
    """
    Describes layer patterns in model's graph that should be considered as a single node
    during quantizer arrangement search algorithm

    :param _graph: Graph contains layer pattern/patterns
    """
    NODE_COUNTER = 0
    INPUT_NODE_TYPE = 'INPUT_NODE'

    def __init__(self, label: str, types: Optional[Union[str, List[str]]] = None):
        """
        :param types: List or signle string of backend operations names
         that should be considered as one single node
        """
        self._graph = nx.DiGraph()
        self.name = label
        if types is not None:
            if isinstance(types, str):
                types = [types]
            self._graph.add_node(GraphPattern.NODE_COUNTER, type=types, label=label)
            GraphPattern.NODE_COUNTER += 1

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
            # Special case whtn first node is INPUT_NODE_TYPE
            if second_graph.nodes[first_node_second_graph]['type'][0] == GraphPattern.INPUT_NODE_TYPE:
                successors = union_graph.successors(first_node_second_graph)
                new_edges = list(it.product([last_node_first_graph], successors))
                union_graph.add_edges_from(new_edges)
                union_graph.remove_node(first_node_second_graph)
            else:
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

    def __or__(self, other: 'GraphPattern') -> 'GraphPattern':
        """
        Add other's DiGraph nodes to self's DiGraph as a new weakly connected components.
        It is a syntax sugar of 'add_pattern_alternative()'

        :param other: GraphPattern that will be added
        """
        new_pattern = copy.deepcopy(self)
        other_copy = create_copy_of_subgraph(other.graph)
        add_subgraph_to_graph(new_pattern._graph, other_copy)
        return new_pattern

    def __eq__(self, other: 'GraphPattern') -> bool:
        return ism.is_isomorphic(self.graph, other.graph)

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph

    def add_pattern_alternative(self, other: 'GraphPattern') -> None:
        """
        Add 'other' pattern as a connected weakly component to 'self' pattern.

        :param other: GraphPattern that will be added
        """
        other_graph = other.graph
        self_graph = self.graph
        for node in other_graph.nodes:
            if node in self_graph.nodes:
                assert False
            self_graph.add_node(node, type=other_graph.nodes[node]['type'], label=other_graph.nodes[node]['label'])
        for edge in other_graph.edges:
            self_graph.add_edge(edge[0], edge[1])

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
        if edges is None:
            first_node_other = list(nx.lexicographical_topological_sort(other.graph, key=int))[0]
            assert other.graph.in_degree(first_node_other) == 0
            last_node_self = list(nx.lexicographical_topological_sort(self.graph, key=int))[-1]
            assert self.graph.out_degree(last_node_self) == 0
            edges = [[last_node_self, first_node_other]]
        for edge in edges:
            assert edge[0] in self.graph.nodes
            assert edge[1] in other.graph.nodes
        add_subgraph_to_graph(self.graph, other.graph)
        for edge in edges:
            self.graph.add_edge(edge[0], edge[1])

    def add_node(self, label: str, node_type: Union[str, List[str]]) -> int:
        if isinstance(node_type, str):
            node_type = [node_type]
        self.graph.add_node(GraphPattern.NODE_COUNTER, type=node_type, label=label)
        GraphPattern.NODE_COUNTER += 1
        return GraphPattern.NODE_COUNTER - 1

    def add_edge(self, u_name, v_name) -> None:
        self.graph.add_edge(u_name, v_name)

    def get_weakly_connected_subgraphs(self) -> List[nx.DiGraph]:
        return [self.graph.subgraph(c) for c in nx.weakly_connected_components(self.graph)]

    def dump_graph(self, path: str) -> None:
        nx.drawing.nx_pydot.write_dot(self.graph, path)

    def visualize_graph(self):
        pass
