from tests.common.graph.test_graph_pattern import TestPattern
from nncf.common.graph.graph_matching import find_subgraphs_matching_pattern
import networkx as nx

import itertools


def test_ops_combination_patterns():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_edge('1', '2')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [['1', '2']]

    pattern = TestPattern.first_pattern + TestPattern.second_pattern | TestPattern.third_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_edge('1', '2')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [['1', '2']]

    pattern = (TestPattern.first_pattern + TestPattern.second_pattern)
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_node('3', type='e')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('1', '3')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)

    assert matches == [['1', '2', '3']]


def test_no_matches():
    pattern = (TestPattern.first_pattern + TestPattern.second_pattern + TestPattern.third_pattern)
    pattern_nodes = list(pattern.graph.nodes)
    third_nodes = list(TestPattern.third_pattern.graph.nodes)
    edges = list(itertools.product(pattern_nodes, third_nodes))
    pattern.join_patterns(TestPattern.third_pattern, edges)

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_node('3', type='e')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)

    assert not matches


def test_two_matches():
    pattern = TestPattern.first_pattern + TestPattern.second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_node('3', type='e')
    ref_graph.add_node('4', type='c')
    ref_graph.add_node('5', type='a')
    ref_graph.add_node('6', type='d')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    ref_graph.add_edge('5', '6')

    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    matches.sort()
    assert matches == [['1', '2'], ['5', '6']]


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
        '1': {'type': 'a'}, '2': {'type': 'b'}, '3': {'type': 'c'}, '4': {'type': 'a'}, '5': {'type': 'e'},
        '6': {'type': 'a'}, '7': {'type': 'a'}, '8': {'type': 'b'}, '9': {'type': 'c'}
    }
    for k, attrs in nodes.items():
        ref_graph.add_node(k, **attrs)
    ref_graph.add_edges_from([('1', '2'), ('2', '3'), ('2', '4'), ('4', '5'), ('5', '6'),
                              ('3', '5'), ('7', '1'), ('8', '3'), ('9', '6')])
    return ref_graph


def test_matches_with_non_pattern_node_type():
    pattern = TestPattern.forth_pattern + TestPattern.first_pattern + TestPattern.second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='a')
    ref_graph.add_node('3', type='c')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [['2', '3']]

    pattern = TestPattern.forth_pattern + TestPattern.first_pattern + \
              TestPattern.second_pattern + TestPattern.forth_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='a')
    ref_graph.add_node('3', type='c')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [['2', '3']]

    pattern = TestPattern.pattern_with_non_pattern_nodes

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='b')
    ref_graph.add_node('3', type='c')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='b')
    ref_graph.add_node('4', type='a')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '4')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = create_graph_with_many_nodes()
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [['1', '2', '4', '3', '5', '6']]


def test_matches_with_any_pattern_node_type():
    pattern = TestPattern.pattern_with_any_pattern_nodes

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='b')
    ref_graph.add_node('3', type='c')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='b')
    ref_graph.add_node('4', type='a')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '4')
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert not matches

    ref_graph = create_graph_with_many_nodes()
    matches = find_subgraphs_matching_pattern(ref_graph, pattern)
    assert matches == [['7', '1', '2', '4', '8', '3', '5', '9', '6']]
