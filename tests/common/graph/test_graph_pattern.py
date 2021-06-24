from nncf.common.graph.patterns import GraphPattern
import networkx as nx

import itertools


def test_ops_combination_two_patterns():
    first_type = ['a', 'b']
    second_type = ['c', 'd']

    first_pattern = GraphPattern('first', first_type)
    second_pattern = GraphPattern('second', second_type)

    ref_pattern = GraphPattern('first', first_type)
    added_node = ref_pattern.add_node('second', ['c', 'd'])
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    adding_pattern = first_pattern + second_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern('first', first_type)
    _ = ref_pattern.add_node('second', ['c', 'd'])

    adding_pattern = first_pattern | second_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern('first', first_type)
    added_node = ref_pattern.add_node('second', ['c', 'd'])
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    first_nodes = list(first_pattern.graph.nodes)
    second_nodes = list(second_pattern.graph.nodes)
    edges = list(itertools.product(first_nodes, second_nodes))
    first_pattern.join_patterns(second_pattern, edges)
    assert ref_pattern == first_pattern


def test_ops_combination_three_patterns():
    first_type = ['a', 'b']
    second_type = ['c', 'd']
    third_type = ['e']

    first_pattern = GraphPattern('first', first_type)
    second_pattern = GraphPattern('second', second_type)
    third_pattern = GraphPattern('third', third_type)

    ref_pattern = GraphPattern('first', first_type)
    added_node = ref_pattern.add_node('second', second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    _ = ref_pattern.add_node('third', third_type)

    adding_pattern = first_pattern + second_pattern | third_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern('first', first_type)
    _ = ref_pattern.add_node('second', second_type)
    _ = ref_pattern.add_node('third', third_type)

    adding_pattern = first_pattern | second_pattern | third_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern('first', first_type)
    added_node = ref_pattern.add_node('second', second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    added_node = ref_pattern.add_node('third', third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    pattern = (first_pattern + second_pattern)
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(third_pattern, edges)

    # adding_pattern = (first_pattern + second_pattern) * third_pattern
    assert ref_pattern == pattern

    ref_pattern = GraphPattern('first', first_type)
    added_node = ref_pattern.add_node('second', second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    last_node = list(nx.topological_sort(ref_pattern.graph))[-1]
    added_node = ref_pattern.add_node('third', third_type)
    ref_pattern.add_edge(last_node, added_node)

    added_node = ref_pattern.add_node('third', third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    pattern = (first_pattern + second_pattern + third_pattern)
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(third_pattern, edges)

    # adding_pattern = (first_pattern + second_pattern + third_pattern) * third_pattern
    assert ref_pattern == pattern
