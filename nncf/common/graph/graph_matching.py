# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Set

import networkx as nx
import networkx.algorithms.isomorphism as ism

from nncf.common.graph.patterns import GraphPattern


def is_nodes_degrees_match(
    graph: nx.DiGraph, pattern_graph: nx.DiGraph, mapping: Dict[str, str], first_node: str, last_node: str
):
    """
    Checks amount of input and output edges for each node pairs in isomorphic mapping is matching
    except for precessors of the first node and successors of the last node.
    Isomorphic subgraphs could not have different edges between nodes inside subgraphs,
    but could have connections to other nodes in grpah.
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
    :param pattern_graph: The pattern graph.
    :param mapping: Mapping between graph nodes and pattern graph nodes.
    :param first_node: Node key for starting node in matched subgraph.
    :param last_node: Node key for ending node in matched subgraph.
    :return: True if amount of input and output edges for each node pairs in isomorphic
        mapping is matching except for presestors of the first node and successors of the last node,
        False otherwise.
    """
    for graph_key, pattern_key in mapping.items():
        for attr in ["pred", "succ"]:
            if graph_key == first_node and attr == "pred":
                continue
            if graph_key == last_node and attr == "succ":
                continue

            def _len(_graph, _key):
                return len(getattr(_graph, attr)[_key].keys())

            if not _len(graph, graph_key) == _len(pattern_graph, pattern_key):
                return False
    return True


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
            if attr == GraphPattern.LABEL_ATTR:
                continue
            if attr == GraphPattern.METATYPE_ATTR:
                # GraphPattern.ANY_PATTERN_NODE_TYPE and GraphPattern.NON_PATTERN_NODE_TYPE
                # are matched to any node type.

                if (
                    GraphPattern.ANY_PATTERN_NODE_TYPE in node_2[attr]
                    or GraphPattern.NON_PATTERN_NODE_TYPE in node_2[attr]
                ):
                    continue
                # Torch and TF pattern mapping based on 'type' section,
                # While ONNX mapping based on metatypes -
                # to support all of them, we need to check the existane of the attributes
                if GraphPattern.NODE_TYPE_ATTR in node_1:
                    if node_1[GraphPattern.NODE_TYPE_ATTR] in node_2[attr]:
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

    def sort_patterns(pattern: nx.DiGraph):
        """
        Sort patterns by their length,
        keeping in mind that if node type is GraphPattern.NON_PATTERN_NODE_TYPE it shouldn't count.
        """
        pattern_len = len(pattern)
        for node in pattern.nodes:
            if GraphPattern.NON_PATTERN_NODE_TYPE in pattern_graph.graph.nodes.get(node).get(
                GraphPattern.METATYPE_ATTR, []
            ):
                pattern_len -= 1
        return pattern_len

    # Get all patterns sorted by their lengths
    # as we want match the longest patterns first

    patterns = sorted(patterns, key=sort_patterns, reverse=True)

    for pattern in patterns:
        matcher = ism.DiGraphMatcher(graph, pattern, node_match=are_nodes_matching, edge_match=are_edges_matching)
        for subgraph in matcher.subgraph_isomorphisms_iter():
            # Bottleneck that need to sort by id for result consistency
            pattern_subgraph = list(
                nx.lexicographical_topological_sort(graph.subgraph(subgraph), key=lambda x: int(x.split()[0]))
            )

            # If some nodes are outside the pattern - remove them from pattern_subgraph
            for node, pattern_node_id in matcher.mapping.items():
                pattern_node = pattern_graph.graph.nodes[pattern_node_id]
                pattern_node_types = pattern_node.get(GraphPattern.METATYPE_ATTR, [])
                if GraphPattern.NON_PATTERN_NODE_TYPE in pattern_node_types:
                    pattern_subgraph.remove(node)

            if any(node in visited_nodes for node in pattern_subgraph):
                continue

            if not is_nodes_degrees_match(graph, pattern, matcher.mapping, pattern_subgraph[0], pattern_subgraph[-1]):
                continue

            visited_nodes.update(pattern_subgraph)
            subgraphs.append(pattern_subgraph)

    return subgraphs if subgraphs else []
