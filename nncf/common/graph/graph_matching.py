# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Dict, List

import networkx as nx  # type:ignore
import networkx.algorithms.isomorphism as ism  # type:ignore

from nncf.common.graph.patterns import GraphPattern

ATTRS_TO_SKIP = [GraphPattern.LABEL_ATTR, GraphPattern.PATTERN_NODE_TO_EXCLUDE]


def _are_nodes_matched(node_1, node_2) -> bool:  # type:ignore
    for attr in node_2:
        if attr in ATTRS_TO_SKIP:
            continue
        if attr == GraphPattern.METATYPE_ATTR:
            # GraphPattern.ANY_PATTERN_NODE_TYPE and GraphPattern.NON_PATTERN_NODE_TYPE
            # are matched to any node type.
            if GraphPattern.ANY_PATTERN_NODE_TYPE in node_2[attr] or GraphPattern.NON_PATTERN_NODE_TYPE in node_2[attr]:
                continue
            # Torch and TF pattern mapping based on 'type' section,
            # While ONNX mapping based on metatypes -
            # to support all of them, we need to check the existence of the attributes
            if GraphPattern.NODE_TYPE_ATTR in node_1 and node_1[GraphPattern.NODE_TYPE_ATTR] in node_2[attr]:
                continue
        if node_1[attr] not in node_2[attr]:
            return False
    return True


def _sort_patterns_by_len(pattern: nx.DiGraph) -> int:
    """
    Sort patterns by their length. GraphPattern.NON_PATTERN_NODE_TYPE is not counted as a pattern node.
    """
    non_pattern_nodes = [
        node_id
        for node_id, node_data in pattern.nodes(data=True)
        if GraphPattern.NON_PATTERN_NODE_TYPE in node_data.get(GraphPattern.METATYPE_ATTR, [])
    ]
    return len(pattern) - len(non_pattern_nodes)


def _is_subgraph_matching_strict(graph: nx.DiGraph, pattern: nx.DiGraph, subgraph: Dict[str, str]) -> bool:
    """
    Checks out whether the matched subgraph has:
    1) External predecessors of starting nodes.
    2) External successors of the last nodes.
    3) External successors or predecessors of the nodes which are not starting and last.
    If any of these conditions is True, than returns False, otherwise - True.
    The checks are skipped for NON_PATTERN_NODE_TYPE.
    Example:
    This subgraph matching is not strict.
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
    :param pattern: The matched pattern.
    :param subgraph: A subgraph of the model graph including the nodes outside the pattern.
    :return: If any of three conditions is True than returns False, otherwise - True.
    """
    starting_nodes = []
    last_nodes = []
    for node in pattern.nodes:
        if not pattern.pred[node] and pattern.succ[node]:
            starting_nodes.append(node)
        if pattern.pred[node] and not pattern.succ[node]:
            last_nodes.append(node)

    for node_from_graph, node_from_pattern in subgraph.items():
        if GraphPattern.NON_PATTERN_NODE_TYPE in pattern.nodes[node_from_pattern].get(GraphPattern.METATYPE_ATTR, []):
            continue
        predecessors_keys = graph.pred[node_from_graph].keys()
        successor_keys = graph.succ[node_from_graph].keys()
        has_external_successors = any(successor_key not in subgraph for successor_key in successor_keys)
        has_external_predcessors = any(predecessor_key not in subgraph for predecessor_key in predecessors_keys)
        if node_from_pattern in starting_nodes and has_external_successors:
            return False
        if node_from_pattern in last_nodes and has_external_predcessors:
            return False
        if (node_from_pattern not in last_nodes and node_from_pattern not in starting_nodes) and (
            has_external_successors or has_external_predcessors
        ):
            return False
    return True


def _copy_subgraph_excluding_non_pattern_node(subgraph: Dict[str, str], pattern_graph: GraphPattern) -> Dict[str, str]:
    """
    Copies a matching subgraph excluding the nodes having GraphPattern.NON_PATTERN_NODE_TYPE
       or GraphPattern.PATTERN_NODE_TO_EXCLUDE.

    :param subgraph: Subgraph
    :param pattern_graph: A graph consists of patterns to match.
    :return: New subgraph without excluded nodes.
    """
    output = {}
    for node_from_graph, node_from_pattern in subgraph.items():
        pattern_node = pattern_graph.graph.nodes[node_from_pattern]
        pattern_node_types = pattern_node.get(GraphPattern.METATYPE_ATTR, [])
        if GraphPattern.NON_PATTERN_NODE_TYPE in pattern_node_types:
            continue
        if pattern_node.get(GraphPattern.PATTERN_NODE_TO_EXCLUDE, False):
            continue
        output[node_from_graph] = node_from_pattern

    return output


def find_subgraphs_matching_pattern(
    graph: nx.DiGraph, pattern_graph: GraphPattern, strict: bool = True
) -> List[List[str]]:
    """
    Finds a list of nodes which define a subgraph matched a pattern in pattern_graph.
    Nodes in each subgraph is stored in lexicographical_topological_sort.

    :param graph: The model graph.
    :param pattern_graph: A graph consists of patterns to match.
    :param strict: If True returns only strict matched subgraphs, if False - all matched subgraphs.
    :return: A list of subgraphs are matched to the patterns. Each subgraph is defined as a list of node keys.
    """
    subgraphs = []
    matched_nodes = set()
    patterns = pattern_graph.get_weakly_connected_subgraphs()
    patterns = sorted(patterns, key=_sort_patterns_by_len, reverse=True)
    for pattern in patterns:
        matcher = ism.DiGraphMatcher(graph, pattern, node_match=_are_nodes_matched)
        for subgraph in matcher.subgraph_isomorphisms_iter():
            if strict and not _is_subgraph_matching_strict(graph, pattern, subgraph):
                continue

            subgraph = _copy_subgraph_excluding_non_pattern_node(subgraph, pattern_graph)
            is_any_node_matched = any(node in matched_nodes for node in subgraph)

            if is_any_node_matched:
                continue

            matched_nodes.update(subgraph)
            sorted_nodes_subgraph = list(nx.lexicographical_topological_sort(graph.subgraph(subgraph)))
            subgraphs.append(sorted_nodes_subgraph)

    return subgraphs
