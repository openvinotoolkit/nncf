from nncf.torch.graph.patterns import GraphPattern
from nncf.torch.graph.graph_matching import find_subgraphs_match_expression
import networkx as nx


def test_ops_combination_patterns():
    first_type = ['a', 'b']
    second_type = ['c', 'd']
    third_type = ['e']

    first_pattern = GraphPattern(first_type)
    second_pattern = GraphPattern(second_type)
    third_pattern = GraphPattern(third_type)

    pattern = first_pattern + second_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_edge('1', '2')
    matches = find_subgraphs_match_expression(ref_graph, pattern)
    assert matches == [['1', '2']]

    pattern = first_pattern + second_pattern | third_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_edge('1', '2')
    matches = find_subgraphs_match_expression(ref_graph, pattern)
    assert matches == [['1', '2']]

    pattern = (first_pattern + second_pattern) * third_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_node('3', type='e')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('1', '3')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_match_expression(ref_graph, pattern)

    assert matches == [['1', '2', '3']]


def test_no_mathces():
    first_type = ['a', 'b']
    second_type = ['c', 'd']
    third_type = ['e']

    first_pattern = GraphPattern(first_type)
    second_pattern = GraphPattern(second_type)
    third_pattern = GraphPattern(third_type)

    pattern = (first_pattern + second_pattern + third_pattern) * third_pattern

    ref_graph = nx.DiGraph()
    ref_graph.add_node('1', type='a')
    ref_graph.add_node('2', type='c')
    ref_graph.add_node('3', type='e')
    ref_graph.add_edge('1', '2')
    ref_graph.add_edge('2', '3')
    matches = find_subgraphs_match_expression(ref_graph, pattern)

    assert not matches[0]


def test_two_matches():
    first_type = ['a', 'b']
    second_type = ['c', 'd']

    first_pattern = GraphPattern(first_type)
    second_pattern = GraphPattern(second_type)

    pattern = first_pattern + second_pattern

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

    matches = find_subgraphs_match_expression(ref_graph, pattern)
    assert matches == [['1', '2'], ['5', '6']]
