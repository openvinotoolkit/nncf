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
    return sorted(in_edge_boundary), sorted(out_edge_boundary)  # must be sorted for determinism


def is_subgraph_has_inner_outgoing_edges(graph: nx.DiGraph, full_subgraph_with_non_pattern_nodes: List[str],
                                         pattern_subgraph: List[str]) -> bool:
    """
    Check out whether the 'pattern_subgraph' has outgoing edges which
     aren't connected with nodes from full_subgraph_with_non_pattern_nodes
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
    :param full_subgraph_with_non_pattern_nodes: A subgraph of the model graph with nodes outside the patter.
    :param pattern_subgraph: A subgraph of the model graph
    :return: True if the subgraph contains outgoing edges starting not from the last node,
        False - otherwise.
    """
    first_node = pattern_subgraph[0]
    last_node = pattern_subgraph[-1]
    for node_key in pattern_subgraph:
        if node_key == last_node:
            predecessors = list(graph.pred[node_key].keys())
            if any(predecessor not in full_subgraph_with_non_pattern_nodes for predecessor in predecessors):
                return True
        elif node_key == first_node:
            successors = list(graph.succ[node_key].keys())
            if any(successor not in full_subgraph_with_non_pattern_nodes for successor in successors):
                return True
        else:
            successors = list(graph.succ[node_key].keys())
            predecessors = list(graph.pred[node_key].keys())
            if any(successors_key not in full_subgraph_with_non_pattern_nodes for successors_key in successors):
                return True
            if any(predecessor not in full_subgraph_with_non_pattern_nodes for predecessor in predecessors):
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
            if attr == 'label':
                continue
            if attr == 'type':
                if GraphPattern.ANY_PATTERN_NODE_TYPE in node_2['type'] or \
                        GraphPattern.NON_PATTERN_NODE_TYPE in node_2['type']:
                    continue
            if attr == 'outgoing_edges':
                continue
            if node_1[attr] not in node_2[attr]:
                return False
        # Save matches between pattern nodes and graph nodes

        nonlocal matches
        # Bottleneck: the node must have 'key' attribute

        matches[node_1['key']] = node_2
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
    def sort_patterns(pattern: nx.DiGraph):
        """
        Sort patterns by their length,
        keeping in mind that if node type is GraphPattern.NON_PATTERN_NODE_TYPE it shouldn't count.
        """
        pattern_len = len(pattern)
        for node in pattern.nodes:
            if GraphPattern.NON_PATTERN_NODE_TYPE in pattern_graph.graph.nodes.get(node)['type']:
                pattern_len -= 1
        return pattern_len

    patterns = sorted(patterns, key=sort_patterns, reverse=True)

    for pattern in patterns:
        matcher = ism.DiGraphMatcher(graph, pattern,
                                     node_match=are_nodes_matching,
                                     edge_match=are_edges_matching)
        # Restore matches for every pattern

        matches = {}
        for subgraph in matcher.subgraph_isomorphisms_iter():
            # Bottleneck that need to sort by id for result consistency
            pattern_subgraph = list(nx.lexicographical_topological_sort(graph.subgraph(subgraph),
                                                                        key=lambda x: int(x.split()[0])))
            # Remove matches from unsuccessful matched patterns
            pattern_matches = {}
            for node_name, node_attrs in matches.items():
                if node_name in pattern_subgraph:
                    pattern_matches[node_name] = node_attrs

            full_subgraph_with_non_pattern_nodes = pattern_subgraph[:]
            outside_pattern_nodes = []

            # If some nodes outside the pattern - remove them from pattern_subgraph

            for node_name, node_attrs in pattern_matches.items():
                if GraphPattern.NON_PATTERN_NODE_TYPE in node_attrs['type']:
                    outside_pattern_nodes.append(graph.nodes.get(node_name))
            for node in outside_pattern_nodes:
                pattern_subgraph.remove(node['key'])

            matches = {}
            is_visited_node = any(node in visited_nodes for node in pattern_subgraph)
            if is_visited_node:
                continue
            if is_subgraph_has_inner_outgoing_edges(graph, full_subgraph_with_non_pattern_nodes, pattern_subgraph):
                continue
            visited_nodes.update(pattern_subgraph)
            subgraphs.append(pattern_subgraph)

    return subgraphs if subgraphs else []
