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

import copy
import itertools
from pathlib import Path

import networkx as nx

from nncf.common.graph.patterns import GraphPattern


class TestPattern:
    first_type = ["a", "b"]
    second_type = ["c", "d"]
    third_type = ["e"]
    forth_type = [GraphPattern.NON_PATTERN_NODE_TYPE]
    fifth_type = [GraphPattern.ANY_PATTERN_NODE_TYPE]

    first_pattern = GraphPattern()
    first_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: first_type})
    second_pattern = GraphPattern()
    second_pattern.add_node(**{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: second_type})
    third_pattern = GraphPattern()
    third_pattern.add_node(**{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: third_type})
    forth_pattern = GraphPattern()
    forth_pattern.add_node(**{GraphPattern.LABEL_ATTR: "forth", GraphPattern.METATYPE_ATTR: forth_type})
    fifth_pattern = GraphPattern()
    fifth_pattern.add_node(**{GraphPattern.LABEL_ATTR: "fifth", GraphPattern.METATYPE_ATTR: fifth_pattern})

    # pattern_with_non_pattern_nodes |  pattern_with_any_pattern_nodes
    #        NON                     |            ANY
    #         |                      |             |
    #         1                      |             1
    #         |                      |             |
    #         2  NON                 |             2  ANY
    #        / \ /                   |            / \ /
    #       4   3                    |           4   3
    #       |  /                     |           |  /
    #       | /                      |           | /
    #       |/                       |           |/
    #       5                        |           5
    #       |                        |           |
    #       6---NON                  |           6---ANY

    pattern_with_non_pattern_nodes = GraphPattern()
    pattern_with_any_pattern_nodes = GraphPattern()
    common_nodes = {
        "1": {GraphPattern.METATYPE_ATTR: "a"},
        "2": {GraphPattern.METATYPE_ATTR: "b"},
        "3": {GraphPattern.METATYPE_ATTR: "c"},
        "4": {GraphPattern.METATYPE_ATTR: "a"},
        "5": {GraphPattern.METATYPE_ATTR: "e"},
        "6": {GraphPattern.METATYPE_ATTR: "a"},
    }
    non_pattern_nodes = {
        "7": {GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE},
        "8": {GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE},
        "9": {GraphPattern.METATYPE_ATTR: GraphPattern.NON_PATTERN_NODE_TYPE},
    }
    any_pattern_nodes = {
        "7": {GraphPattern.METATYPE_ATTR: GraphPattern.ANY_PATTERN_NODE_TYPE},
        "8": {GraphPattern.METATYPE_ATTR: GraphPattern.ANY_PATTERN_NODE_TYPE},
        "9": {GraphPattern.METATYPE_ATTR: GraphPattern.ANY_PATTERN_NODE_TYPE},
    }
    label_to_non_pattern_nodes = {}
    label_to_any_pattern_nodes = {}
    for label, attrs in common_nodes.items():
        label_to_non_pattern_nodes[label] = pattern_with_non_pattern_nodes.add_node(label=label, **attrs)
        label_to_any_pattern_nodes[label] = pattern_with_any_pattern_nodes.add_node(label=label, **attrs)
    for label, attrs in non_pattern_nodes.items():
        label_to_non_pattern_nodes[label] = pattern_with_non_pattern_nodes.add_node(label=label, **attrs)
    for label, attrs in any_pattern_nodes.items():
        label_to_any_pattern_nodes[label] = pattern_with_any_pattern_nodes.add_node(label=label, **attrs)

    edges = [("1", "2"), ("2", "3"), ("2", "4"), ("4", "5"), ("5", "6"), ("3", "5"), ("7", "1"), ("8", "3"), ("9", "6")]
    for edge in edges:
        pattern_with_non_pattern_nodes.add_edge(
            label_to_non_pattern_nodes[edge[0]], label_to_non_pattern_nodes[edge[1]]
        )
        pattern_with_any_pattern_nodes.add_edge(
            label_to_any_pattern_nodes[edge[0]], label_to_any_pattern_nodes[edge[1]]
        )


def test_ops_combination_two_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern
    ref_pattern = GraphPattern()
    ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: TestPattern.second_type}
    )
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    assert ref_pattern == pattern

    pattern = TestPattern.first_pattern | TestPattern.second_pattern
    ref_pattern = GraphPattern()
    ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    _ = ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: TestPattern.second_type})
    assert ref_pattern == pattern


def test_ops_combination_three_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern | TestPattern.third_pattern
    ref_pattern = GraphPattern()
    ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    added_node = ref_pattern.add_node(label="second", type=TestPattern.second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    _ = ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: TestPattern.third_type})
    assert ref_pattern == pattern

    pattern = TestPattern.first_pattern | TestPattern.second_pattern | TestPattern.third_pattern
    ref_pattern = GraphPattern()
    _ = ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    _ = ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: TestPattern.second_type})
    _ = ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: TestPattern.third_type})
    assert ref_pattern == pattern

    pattern = TestPattern.first_pattern + TestPattern.second_pattern
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)
    ref_pattern = GraphPattern()
    ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: TestPattern.second_type}
    )
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: TestPattern.third_type}
    )
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    assert ref_pattern == pattern


def test_join_patterns_func():
    ref_pattern = GraphPattern()
    ref_pattern.add_node(label="first", type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label="second", type=TestPattern.second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    first_nodes = list(TestPattern.first_pattern.graph.nodes)
    second_nodes = list(TestPattern.second_pattern.graph.nodes)
    edges = list(itertools.product(first_nodes, second_nodes))
    pattern = copy.copy(TestPattern.first_pattern)
    pattern.join_patterns(TestPattern.second_pattern, edges)
    assert ref_pattern == pattern


def test_join_patterns_func_three_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern + TestPattern.third_pattern
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)
    ref_pattern = GraphPattern()
    _ = ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: TestPattern.second_type}
    )
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    last_node = list(nx.topological_sort(ref_pattern.graph))[-1]
    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: TestPattern.third_type}
    )
    ref_pattern.add_edge(last_node, added_node)

    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: TestPattern.third_type}
    )
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    assert ref_pattern == pattern


def test_join_pattern_with_special_input_node():
    pattern = TestPattern.first_pattern
    second_pattern = GraphPattern()
    second_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "second", GraphPattern.METATYPE_ATTR: GraphPattern.ANY_PATTERN_NODE_TYPE}
    )
    pattern.join_patterns(second_pattern)
    pattern.join_patterns(TestPattern.third_pattern)

    ref_pattern = GraphPattern()
    ref_pattern.add_node(**{GraphPattern.LABEL_ATTR: "first", GraphPattern.METATYPE_ATTR: TestPattern.first_type})
    added_node = ref_pattern.add_node(
        **{GraphPattern.LABEL_ATTR: "third", GraphPattern.METATYPE_ATTR: TestPattern.third_type}
    )
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    assert pattern == ref_pattern


def test_dump(tmp_path: Path):
    path_dot = tmp_path / "pattern.dot"
    TestPattern.first_pattern.dump_graph(path_dot)
    assert path_dot.is_file()
    path_dot.unlink()
