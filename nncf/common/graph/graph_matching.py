"""
 Copyright (c) 2019 Intel Corporation
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

from typing import List
import networkx as nx
import networkx.algorithms.isomorphism as ism
from nncf.torch.graph.patterns import GraphPattern


def find_whether_subgraph_has_inner_outgoing_edges(graph: nx.DiGraph, subgraph: List[str]) -> bool:
    """
    Check out whether the subgraph has outgoing edges starting not from the last node.
    Example:
    (conv2d + BN + ReLU pattern):
            ...
             |
          (conv2d)
             |------\
            (BN)    |
             |      |
           (RELU)   |
             |      |
           (cat)----/
             |
            ...
    :param graph: The model graph.
    :param subgraph: A subgraph of the model graph.
    :return: True if the subgraph contains outgoing edges starting not from the last node,
        False - otherwise.
    """
    for node_key in subgraph[:-1]:
        successors = list(graph.succ[node_key].keys())
        for successors_key in successors:
            if successors_key not in subgraph:
                return True

    # Breaking input edges
    for node_key in subgraph[1:]:
        predecessors = list(graph.pred[node_key].keys())
        for predecessors_key in predecessors:
            if predecessors_key not in subgraph:
                return True
    return False


def find_subgraphs_matching_expression(graph: nx.DiGraph, expression: Expression) -> List[List[str]]:
    """
    Find a list of subgraphs for the particular graph that match the pattern expression.
    :param graph: The model graph.
    :param pattern_graph: A graph consists of patterns for layer fusing logic.
    :return: A list of subgraphs for the particular graph, matching the pattern expression.
    """

    def are_nodes_matching(node_1, node_2):
        return node_1['type'] in node_2['type']

    subgraphs = []
    visited_nodes = set()
    # Get all patterns sorted by their lengths
    patterns = []
    for c in nx.weakly_connected_components(pattern_graph._graph):
        patterns.append(pattern_graph._graph.subgraph(c))
    patterns = sorted(patterns, key=lambda x: len(x), reverse=True)
    for pattern in patterns:
        matcher = ism.DiGraphMatcher(graph, pattern, node_match=are_nodes_matching)
        gas = list(matcher.subgraph_isomorphisms_iter())
        for subgraph in gas:
            subgraph = list(subgraph)
            is_visited_node = any(node in visited_nodes for node in subgraph)
            if is_visited_node:
                continue
            if find_whether_subgraph_has_inner_outgoing_edges(graph, subgraph):
                continue
            visited_nodes.update(subgraph)
            subgraphs.append(subgraph)

    return subgraphs
