from nncf.common.graph.patterns import GraphPattern
import networkx as nx

import copy
import itertools


class TestPattern:
    first_type = ['a', 'b']
    second_type = ['c', 'd']
    third_type = ['e']
    forth_type = [GraphPattern.NON_PATTERN_NODE_TYPE]
    fifth_type = [GraphPattern.ANY_PATTERN_NODE_TYPE]

    first_pattern = GraphPattern()
    first_pattern.add_node(label='first', type=first_type)
    second_pattern = GraphPattern()
    second_pattern.add_node(label='second', type=second_type)
    third_pattern = GraphPattern()
    third_pattern.add_node(label='third', type=third_type)
    forth_pattern = GraphPattern()
    forth_pattern.add_node(label='forth', type=forth_type)
    fifth_pattern = GraphPattern()
    fifth_pattern.add_node(label='fifth', type=fifth_pattern)

    # pattern_with_non_pattern_nodes
    #        NON
    #         |
    #         1
    #         |
    #         2  NON
    #        / \ /
    #       4   3
    #       |  /
    #       | /
    #       |/
    #       5
    #       |
    #       6---NON

    pattern_with_non_pattern_nodes = GraphPattern()
    first = pattern_with_non_pattern_nodes.add_node(label='1', type=['a'])
    second = pattern_with_non_pattern_nodes.add_node(label='2', type=['b'])
    third = pattern_with_non_pattern_nodes.add_node(label='3', type=['c'])
    forth = pattern_with_non_pattern_nodes.add_node(label='4', type=['a'])
    fifth = pattern_with_non_pattern_nodes.add_node(label='5', type=['e'])
    sixth = pattern_with_non_pattern_nodes.add_node(label='6', type=['a'])
    seventh = pattern_with_non_pattern_nodes.add_node(label='7', type=[GraphPattern.NON_PATTERN_NODE_TYPE])
    eighth = pattern_with_non_pattern_nodes.add_node(label='8', type=[GraphPattern.NON_PATTERN_NODE_TYPE])
    nineth = pattern_with_non_pattern_nodes.add_node(label='9', type=[GraphPattern.NON_PATTERN_NODE_TYPE])
    pattern_with_non_pattern_nodes.add_edge(first, second)
    pattern_with_non_pattern_nodes.add_edge(second, third)
    pattern_with_non_pattern_nodes.add_edge(second, forth)
    pattern_with_non_pattern_nodes.add_edge(forth, fifth)
    pattern_with_non_pattern_nodes.add_edge(third, fifth)
    pattern_with_non_pattern_nodes.add_edge(fifth, sixth)
    pattern_with_non_pattern_nodes.add_edge(seventh, first)
    pattern_with_non_pattern_nodes.add_edge(eighth, third)
    pattern_with_non_pattern_nodes.add_edge(nineth, sixth)

    # pattern_with_any_pattern_nodes
    #        ANY
    #         |
    #         1
    #         |
    #         2  ANY
    #        / \ /
    #       4   3
    #       |  /
    #       | /
    #       |/
    #       5
    #       |
    #       6---ANY

    pattern_with_any_pattern_nodes = GraphPattern()
    first = pattern_with_any_pattern_nodes.add_node(label='1', type=['a'])
    second = pattern_with_any_pattern_nodes.add_node(label='2', type=['b'])
    third = pattern_with_any_pattern_nodes.add_node(label='3', type=['c'])
    forth = pattern_with_any_pattern_nodes.add_node(label='4', type=['a'])
    fifth = pattern_with_any_pattern_nodes.add_node(label='5', type=['e'])
    sixth = pattern_with_any_pattern_nodes.add_node(label='6', type=['a'])
    seventh = pattern_with_any_pattern_nodes.add_node(label='7', type=[GraphPattern.ANY_PATTERN_NODE_TYPE])
    eighth = pattern_with_any_pattern_nodes.add_node(label='8', type=[GraphPattern.ANY_PATTERN_NODE_TYPE])
    nineth = pattern_with_any_pattern_nodes.add_node(label='9', type=[GraphPattern.ANY_PATTERN_NODE_TYPE])
    pattern_with_any_pattern_nodes.add_edge(first, second)
    pattern_with_any_pattern_nodes.add_edge(second, third)
    pattern_with_any_pattern_nodes.add_edge(second, forth)
    pattern_with_any_pattern_nodes.add_edge(forth, fifth)
    pattern_with_any_pattern_nodes.add_edge(third, fifth)
    pattern_with_any_pattern_nodes.add_edge(fifth, sixth)
    pattern_with_any_pattern_nodes.add_edge(seventh, first)
    pattern_with_any_pattern_nodes.add_edge(eighth, third)
    pattern_with_any_pattern_nodes.add_edge(nineth, sixth)


def test_ops_combination_two_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern
    ref_pattern = GraphPattern()
    ref_pattern.add_node(label='first', type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label='second', type=TestPattern.second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    assert ref_pattern == pattern

    pattern = TestPattern.first_pattern | TestPattern.second_pattern
    ref_pattern = GraphPattern()
    ref_pattern.add_node(label='first', type=TestPattern.first_type)
    _ = ref_pattern.add_node(label='second', type=TestPattern.second_type)
    assert ref_pattern == pattern


def test_ops_combination_three_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern | TestPattern.third_pattern
    ref_pattern = GraphPattern()
    ref_pattern.add_node(label='first', type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label='second', type=TestPattern.second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    _ = ref_pattern.add_node(label='third', type=TestPattern.third_type)
    assert ref_pattern == pattern

    pattern = TestPattern.first_pattern | TestPattern.second_pattern | TestPattern.third_pattern
    ref_pattern = GraphPattern()
    _ = ref_pattern.add_node(label='first', type=TestPattern.first_type)
    _ = ref_pattern.add_node(label='second', type=TestPattern.second_type)
    _ = ref_pattern.add_node(label='third', type=TestPattern.third_type)
    assert ref_pattern == pattern

    pattern = (TestPattern.first_pattern + TestPattern.second_pattern)
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)
    ref_pattern = GraphPattern()
    ref_pattern.add_node(label='second', type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label='second', type=TestPattern.second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    added_node = ref_pattern.add_node(label='third', type=TestPattern.third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    assert ref_pattern == pattern


def test_join_patterns_func():
    ref_pattern = GraphPattern()
    ref_pattern.add_node(label='first', type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label='second', type=TestPattern.second_type)
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
    pattern = (TestPattern.first_pattern + TestPattern.second_pattern + TestPattern.third_pattern)
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)
    ref_pattern = GraphPattern()
    _ = ref_pattern.add_node(label='first', type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label='second', type=TestPattern.second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    last_node = list(nx.topological_sort(ref_pattern.graph))[-1]
    added_node = ref_pattern.add_node(label='third', type=TestPattern.third_type)
    ref_pattern.add_edge(last_node, added_node)

    added_node = ref_pattern.add_node(label='third', type=TestPattern.third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    assert ref_pattern == pattern


def test_join_pattern_with_special_input_node():
    pattern = TestPattern.first_pattern
    second_pattern = GraphPattern()
    second_pattern.add_node(label='second', type=GraphPattern.ANY_PATTERN_NODE_TYPE)
    pattern.join_patterns(second_pattern)
    pattern.join_patterns(TestPattern.third_pattern)

    ref_pattern = GraphPattern()
    ref_pattern.add_node(label='first', type=TestPattern.first_type)
    added_node = ref_pattern.add_node(label='third', type=TestPattern.third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    assert pattern == ref_pattern
