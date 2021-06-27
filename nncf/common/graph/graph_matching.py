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
from typing import Set
import networkx as nx
import networkx.algorithms.isomorphism as ism
from nncf.common.graph.patterns import GraphPattern


def get_edge_boundaries(match: List[str], graph: nx.DiGraph):
    out_edge_boundary = list(nx.edge_boundary(graph, match, data=True))
    complement = list(filter(lambda x: x not in match, graph.nodes.keys()))
    in_edge_boundary = list(nx.edge_boundary(graph, complement, data=True))
    return sorted(in_edge_boundary), sorted(out_edge_boundary)


def is_subgraph_has_inner_outgoing_edges(graph: nx.DiGraph, subgraph: List[str]) -> bool:
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
    first_node = subgraph[0]
    last_node = subgraph[-1]
    for node_key in subgraph:
        if node_key == last_node:
            predecessors = list(graph.pred[node_key].keys())
            if any(predecessor not in subgraph for predecessor in predecessors):
                return True
        elif node_key == first_node:
            successors = list(graph.succ[node_key].keys())
            if any(successor not in subgraph for successor in successors):
                return True
        else:
            successors = list(graph.succ[node_key].keys())
            predecessors = list(graph.pred[node_key].keys())
            if any(successors_key not in subgraph for successors_key in successors):
                return True
            if any(predecessor not in subgraph for predecessor in predecessors):
                return True
    return False


def find_subgraphs_matching_pattern(graph: nx.DiGraph, pattern_graph: GraphPattern) -> List[List[str]]:
    """
    Find a list of subgraphs for the particular graph that match the pattern expression.
    :param graph: The model graph.
    :param pattern_graph: A graph consists of patterns for layer fusing logic.
    :return: A list of subgraphs, matching the pattern expression.
        Each subgraph is defined as a list of node keys.
    """

    def are_nodes_matching(node_1, node_2):
        for attr in node_2:
            # Special case for Input node
            if attr == 'type' and node_1['type'] == GraphPattern.PATTERN_INPUT_NODE_TYPE or \
                    attr == 'label':
                continue
            if node_1[attr] not in node_2[attr]:
                return False
        return True

    def are_edges_matching(edge_1, edge_2):
        for attr in edge_2:
            if edge_1[attr] not in edge_2[attr]:
                return False
        return True

    subgraphs = []  # type: List[List[str]]
    visited_nodes = set()  # type: Set[str]
    patterns = []  # type: List[nx.DiGraph]
    for c in nx.weakly_connected_components(pattern_graph.graph):
        patterns.append(pattern_graph.graph.subgraph(c))
    # Get all patterns sorted by their lengths
    # as we want match the longest patterns first
    patterns = sorted(patterns, key=len, reverse=True)
    for pattern in patterns:
        matcher = ism.DiGraphMatcher(graph, pattern,
                                     node_match=are_nodes_matching,
                                     edge_match=are_edges_matching)
        for subgraph in matcher.subgraph_isomorphisms_iter():
            # Bottleneck that need to sort by id for result consistency
            subgraph = list(nx.lexicographical_topological_sort(graph.subgraph(subgraph),
                                                                key=lambda x: int(x.split()[0])))
            is_visited_node = any(node in visited_nodes for node in subgraph)
            if is_visited_node:
                continue
            if is_subgraph_has_inner_outgoing_edges(graph, subgraph):
                continue
            visited_nodes.update(subgraph)
            subgraphs.append(subgraph)

    subgraphs.sort()
    return subgraphs if subgraphs else []
