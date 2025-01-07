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

import itertools

import networkx as nx

from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
from nncf.common.graph.patterns import GraphPattern
from tests.common.graph.test_graph_pattern import TestPattern


def test_ops_combination_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("1", "2")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["1", "2"]]

    pattern = TestPattern.first_pattern + TestPattern.second_pattern | TestPattern.third_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("1", "2")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["1", "2"]]

    pattern = TestPattern.first_pattern + TestPattern.second_pattern
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "e"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("1", "3")
    ref_graph.add_edge("2", "3")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)

    assert matches == [["1", "2", "3"]]


def test_no_matches():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern + TestPattern.third_pattern
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "e"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)

    assert not matches


def test_two_matches():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "e"})
    ref_graph.add_node("4", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_node("5", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("6", **{GraphPattern.METATYPE_ATTR: "d"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    ref_graph.add_edge("5", "6")

    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    matches.sort()
    assert matches == [["1", "2"], ["5", "6"]]


def create_graph_with_many_nodes():
    #     ref_graph
    #         a
    #         |
    #         a
    #         |
    #         b   b
    #        / \ /
    #       a   c
    #       |  /
    #       | /
    #       |/
    #       e
    #       |
    #       a---c

    ref_graph = nx.DiGraph()
    nodes = {
        "1": {GraphPattern.METATYPE_ATTR: "a"},
        "2": {GraphPattern.METATYPE_ATTR: "b"},
        "3": {GraphPattern.METATYPE_ATTR: "c"},
        "4": {GraphPattern.METATYPE_ATTR: "a"},
        "5": {GraphPattern.METATYPE_ATTR: "e"},
        "6": {GraphPattern.METATYPE_ATTR: "a"},
        "7": {GraphPattern.METATYPE_ATTR: "a"},
        "8": {GraphPattern.METATYPE_ATTR: "b"},
        "9": {GraphPattern.METATYPE_ATTR: "c"},
    }
    for k, attrs in nodes.items():
        ref_graph.add_node(k, **attrs)
    ref_graph.add_edges_from(
        [("1", "2"), ("2", "3"), ("2", "4"), ("4", "5"), ("5", "6"), ("3", "5"), ("7", "1"), ("8", "3"), ("9", "6")]
    )
    return ref_graph


def test_matches_with_non_pattern_node_type():
    pattern = TestPattern.forth_pattern + TestPattern.first_pattern + TestPattern.second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["2", "3"]]

    pattern = (
        TestPattern.forth_pattern + TestPattern.first_pattern + TestPattern.second_pattern + TestPattern.forth_pattern
    )

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["2", "3"]]

    pattern = TestPattern.pattern_with_non_pattern_nodes

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "b"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "b"})
    ref_graph.add_node("4", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "4")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = create_graph_with_many_nodes()
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["1", "2", "3", "4", "5", "6"]]


def test_matches_with_any_pattern_node_type():
    pattern = TestPattern.pattern_with_any_pattern_nodes

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "b"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = nx.DiGraph()
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "b"})
    ref_graph.add_node("4", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "4")
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = create_graph_with_many_nodes()
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["7", "1", "2", "4", "8", "3", "5", "9", "6"]]


def test_not_match_edges_inside_pattern():
    ref_graph = nx.DiGraph()
    ref_graph.add_node("0", **{GraphPattern.METATYPE_ATTR: "0"})
    ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a"})
    ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "b"})
    ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "c"})
    ref_graph.add_edge("0", "1")
    ref_graph.add_edge("1", "2")
    ref_graph.add_edge("2", "3")
    ref_graph.add_edge("1", "3")

    pattern = GraphPattern()
    node_1 = pattern.add_node(**{GraphPattern.METATYPE_ATTR: "a"})
    node_2 = pattern.add_node(**{GraphPattern.METATYPE_ATTR: "b"})
    node_3 = pattern.add_node(**{GraphPattern.METATYPE_ATTR: "c"})
    pattern.add_edge(node_1, node_2)
    pattern.add_edge(node_2, node_3)
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    pattern.add_edge(node_1, node_3)
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [["1", "2", "3"]]


def test_non_pattern_graph_with_type():
    for match in [False, True]:
        ref_graph = nx.DiGraph()
        ref_graph.add_node("0", **{GraphPattern.METATYPE_ATTR: "0"})
        ref_graph.add_node("1", **{GraphPattern.METATYPE_ATTR: "a" if match else "0"})
        ref_graph.add_node("2", **{GraphPattern.METATYPE_ATTR: "b"})
        ref_graph.add_node("3", **{GraphPattern.METATYPE_ATTR: "c"})
        ref_graph.add_edge("0", "1")
        ref_graph.add_edge("1", "2")
        ref_graph.add_edge("2", "3")

        pattern = GraphPattern()
        node_1 = pattern.add_node(**{GraphPattern.METATYPE_ATTR: "a", GraphPattern.PATTERN_NODE_TO_EXCLUDE: True})
        node_2 = pattern.add_node(**{GraphPattern.METATYPE_ATTR: "b"})
        node_3 = pattern.add_node(**{GraphPattern.METATYPE_ATTR: "c"})
        pattern.add_edge(node_1, node_2)
        pattern.add_edge(node_2, node_3)

        matches = find_subgraphs_matching_pattern(ref_graph, pattern)
        if not match:
            assert not matches
        else:
            assert matches == [["2", "3"]]
