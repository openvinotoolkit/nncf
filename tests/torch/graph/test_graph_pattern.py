from nncf.torch.graph.patterns import GraphPattern
import networkx as nx


def test_ops_combination_two_patterns():
    first_type = ['a', 'b']
    second_type = ['c', 'd']

    first_pattern = GraphPattern(first_type)
    second_pattern = GraphPattern(second_type)

    ref_pattern = GraphPattern(first_type)
    added_node = ref_pattern.add_node(['c', 'd'])
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    adding_pattern = first_pattern + second_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern(first_type)
    _ = ref_pattern.add_node(['c', 'd'])

    adding_pattern = first_pattern | second_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern(first_type)
    added_node = ref_pattern.add_node(['c', 'd'])
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    adding_pattern = first_pattern * second_pattern
    assert ref_pattern == adding_pattern


def test_ops_combination_three_patterns():
    first_type = ['a', 'b']
    second_type = ['c', 'd']
    third_type = ['e']

    first_pattern = GraphPattern(first_type)
    second_pattern = GraphPattern(second_type)
    third_pattern = GraphPattern(third_type)

    ref_pattern = GraphPattern(first_type)
    added_node = ref_pattern.add_node(second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    _ = ref_pattern.add_node(third_type)

    adding_pattern = first_pattern + second_pattern | third_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern(first_type)
    _ = ref_pattern.add_node(second_type)
    _ = ref_pattern.add_node(third_type)

    adding_pattern = first_pattern | second_pattern | third_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern(first_type)
    added_node = ref_pattern.add_node(second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    added_node = ref_pattern.add_node(third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    adding_pattern = (first_pattern + second_pattern) * third_pattern
    assert ref_pattern == adding_pattern

    ref_pattern = GraphPattern(first_type)
    added_node = ref_pattern.add_node(second_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)
    last_node = list(nx.topological_sort(ref_pattern.graph))[-1]
    added_node = ref_pattern.add_node(third_type)
    ref_pattern.add_edge(last_node, added_node)

    added_node = ref_pattern.add_node(third_type)
    for node in ref_pattern.graph.nodes:
        if node != added_node:
            ref_pattern.add_edge(node, added_node)

    adding_pattern = (first_pattern + second_pattern + third_pattern) * third_pattern
    assert ref_pattern == adding_pattern
