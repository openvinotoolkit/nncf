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

import copy

import networkx as nx
import networkx.algorithms.isomorphism as ism


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
        def _add_second_subgraph_to_first_with_connected_edge(first: nx.DiGraph, second: nx.DiGraph) -> nx.DiGraph:
            union_graph = nx.union(first, second)

            first_nodes = list(nx.topological_sort(first))
            last_node_subgraph = first_nodes[-1]
            second_nodes = list(nx.topological_sort(second))
            first_node_second = second_nodes[0]

            union_graph.add_edge(last_node_subgraph, first_node_second)
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
                subgraph_copy = GraphPattern.create_copy_of_subgraph(self_subgraph)
                other_subgraph_copy = GraphPattern.create_copy_of_subgraph(other_subgraph)
                subgraph_copy = _add_second_subgraph_to_first_with_connected_edge(subgraph_copy, other_subgraph_copy)
                GraphPattern.add_subgraph_to_graph(final_graph, subgraph_copy)

        final_pattern = copy.deepcopy(self)
        final_pattern._graph = final_graph
        return final_pattern

    def __mul__(self, other):
        # all nodes connected with other
        new_pattern = copy.deepcopy(self)
        new_graph = nx.compose(self.graph, other.graph)
        for self_node in self.graph.nodes:
            for other_node in other.graph.nodes:
                new_graph.add_edge(self_node, other_node)
        new_pattern._graph = new_graph
        return new_pattern

    def __or__(self, other):
        new_pattern = copy.deepcopy(self)
        other_copy = GraphPattern.create_copy_of_subgraph(other.graph)
        GraphPattern.add_subgraph_to_graph(new_pattern._graph, other_copy)
        return new_pattern

    def __eq__(self, other):
        return ism.is_isomorphic(self.graph, other.graph)

    @staticmethod
    def create_copy_of_subgraph(subgraph: nx.DiGraph) -> nx.DiGraph:
        mapping = {}
        for node in subgraph.nodes:
            new_node = GraphPattern.NODE_COUNTER
            mapping[node] = new_node
            GraphPattern.NODE_COUNTER += 1
        return nx.relabel_nodes(subgraph, mapping, copy=True)

    @staticmethod
    def add_subgraph_to_graph(graph: nx.DiGraph, subgraph: nx.DiGraph) -> None:
        for node in subgraph.nodes:
            if node in graph.nodes:
                assert False
            graph.add_node(node, type=subgraph.nodes[node]['type'])
        for edge in subgraph.edges:
            graph.add_edge(edge[0], edge[1])

    @property
    def graph(self):
        return self._graph

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


def create_graph_pattern_from_pattern_view(pattern_view: List[str]) -> GraphPattern:
    def is_node_expression(expression: str):
        if "->" not in expression:
            return True
        return False

    def is_edge_expression(expression: str):
        return not is_node_expression(expression)

    def parse_node_str(node: str):
        id_num = node.split()[0]
        start_index = node.find('[')
        types = node[start_index + 1: -1]
        types = types.split(',')
        return id_num, types

    def parse_edge_str(edge: str):
        edge = edge.replace(" ", "")
        out_node, in_node = edge.split('->')
        return out_node, in_node

    graph_pattern = GraphPattern()
    mapping_config_name_pattern_names = {}
    for single_exp in pattern_view:
        if is_node_expression(single_exp):
            id_name, types = parse_node_str(single_exp)
            node_num = graph_pattern.add_node(types)
            mapping_config_name_pattern_names[id_name] = node_num
        elif is_edge_expression(single_exp):
            u_node, v_node = parse_edge_str(single_exp)
            u_name = mapping_config_name_pattern_names[u_node]
            v_name = mapping_config_name_pattern_names[v_node]
            graph_pattern.add_edge(u_name, v_name)
    return graph_pattern
