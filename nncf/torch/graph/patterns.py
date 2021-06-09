"""
 Copyright (c) 2019-2020 Intel Corporation
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

import copy
from nncf.torch.graph.version_agnostic_op_names import VersionAgnosticNames

import networkx as nx
import networkx.algorithms.isomorphism as ism

NODE_COUNTER = 0


class GraphPattern:
    def __init__(self, pattern_name, types):
        global NODE_COUNTER
        self.pattern_name = pattern_name
        self._graph = nx.DiGraph()
        self._graph.add_node(NODE_COUNTER, type=types)
        NODE_COUNTER += 1

    def __add__(self, other):
        def _add_subgraph_to_graph(graph: nx.DiGraph, subgraph: nx.DiGraph) -> None:
            for node in subgraph.nodes:
                if node in graph.nodes:
                    assert False
                graph.add_node(node, type=subgraph.nodes[node]['type'])
            for edge in subgraph.edges:
                graph.add_edge(edge[0], edge[1])

        def _merge_one_subgraph_to_other(main: nx.DiGraph, other: nx.DiGraph) -> nx.DiGraph:
            nodes = list(nx.topological_sort(main))
            last_node_subgraph = nodes[-1]

            for node in other.nodes:
                main.add_node(node, type=other.nodes[node]['type'])
            for edge in other.edges:
                main.add_edge(edge[0], edge[1])

            nodes = list(nx.topological_sort(other))
            first_node_other = nodes[0]

            main.add_edge(last_node_subgraph, first_node_other)
            return main

        def _create_copy_of_subgraph(subgraph: nx.DiGraph) -> nx.DiGraph:
            global NODE_COUNTER
            mapping = {}
            for node in subgraph.nodes:
                new_node = NODE_COUNTER
                mapping[node] = new_node
                NODE_COUNTER += 1
            return nx.relabel_nodes(subgraph, mapping, copy=True)

        weakly_self_subgraphs = self.get_weakly_connected_subgraphs()
        weakly_other_subgraphs = other.get_weakly_connected_subgraphs()
        res_graph = nx.DiGraph()
        for self_subgraph in weakly_self_subgraphs:
            for other_subgraph in weakly_other_subgraphs:
                # Create a copy of subgraph and add to a graph
                subgraph_copy = _create_copy_of_subgraph(self_subgraph)
                other_subgraph_copy = _create_copy_of_subgraph(other_subgraph)
                subgraph_copy = _merge_one_subgraph_to_other(subgraph_copy, other_subgraph_copy)
                _add_subgraph_to_graph(res_graph, subgraph_copy)

        res = copy.copy(self)
        res._graph = res_graph
        return res

    def add_node(self, node_name, t):
        self._graph.add_node(node_name, type=t)

    def add_edge(self, u_name, v_name):
        self._graph.add_edge(u_name, v_name)

    def get_weakly_connected_subgraphs(self):
        return [self._graph.subgraph(c) for c in nx.weakly_connected_components(self._graph)]

    def __eq__(self, other):
        return ism.is_isomorphic(self._graph, other.graph)

    def __or__(self, other):
        self._graph = nx.compose(self._graph, other._graph)
        return self

    def dump_graph(self, path: str):
        nx.drawing.nx_pydot.write_dot(self._graph, path)


LINEAR_OPS_type = ['linear', 'conv2d', 'conv_transpose2d', 'conv3d',
                   'conv_transpose3d', 'conv1d', 'addmm']
RELU_type = ['relu', 'relu_', 'hardtanh']
BN_type = ['batch_norm', 'batch_norm3d']
POOLING_type = ['adaptive_avg_pool2d', 'adaptive_avg_pool3d', 'avg_pool2d', 'avg_pool3d']
NON_RELU_ACTIVATIONS_type = ['elu', 'elu_', 'prelu', 'sigmoid', 'gelu']
ARITHMETIC_type = ['__iadd__', '__add__', '__mul__', '__rmul__']
RELU_graph = GraphPattern('RELU', RELU_type)
BN_graph = GraphPattern('BN', BN_type)
ACTIVATIONS_graph = GraphPattern('ACTIVATIONS', RELU_type + NON_RELU_ACTIVATIONS_type)
ANY_BN_ACT_COMBO_graph = BN_graph + ACTIVATIONS_graph | ACTIVATIONS_graph + BN_graph | BN_graph | ACTIVATIONS_graph
LINEAR_OPS_graph = GraphPattern('LINEAR', LINEAR_OPS_type)
ELTWISE_UNIFORM_OPS_graph = BN_graph | RELU_graph | ACTIVATIONS_graph
ARITHMETIC_graph = GraphPattern('ARITHMETIC', ARITHMETIC_type)

FULL_PATTERN_GRAPH = LINEAR_OPS_graph + ANY_BN_ACT_COMBO_graph | ANY_BN_ACT_COMBO_graph | \
                     ARITHMETIC_graph + ANY_BN_ACT_COMBO_graph | LINEAR_OPS_graph + ELTWISE_UNIFORM_OPS_graph
