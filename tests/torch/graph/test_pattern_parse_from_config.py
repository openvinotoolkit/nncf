from typing import List
import pytest

from nncf.torch.graph.patterns import PATTERN_FACTORY
from nncf.common.graph.graph_matching import find_subgraphs_matching_expression

import networkx as nx


def create_graph_pattern_from_pattern_view(pattern_view: List[str]) -> nx.DiGraph:
    def is_node_expression(expression: str):
        if "->" not in expression:
            return True
        return False

    def is_edge_expression(expression: str):
        return not is_node_expression(expression)

    def parse_node_str(node: str):
        id_num = node.split()[0]
        # start_index = node.find('[')

        op_type = node.split()[1]
        # types = types.split(',')
        return id_num, op_type

    def parse_edge_str(edge: str):
        edge = edge.replace(" ", "")
        out_node, in_node = edge.split('->')
        return out_node, in_node

    graph_pattern = nx.DiGraph()
    for single_exp in pattern_view:
        if is_node_expression(single_exp):
            id_name, types = parse_node_str(single_exp)
            graph_pattern.add_node(str(id_name) + ' ' + str(id_name), type=types)
        elif is_edge_expression(single_exp):
            u_node, v_node = parse_edge_str(single_exp)
            graph_pattern.add_edge(str(u_node) + ' ' + str(u_node), str(v_node) + ' ' + str(v_node))
    return graph_pattern


TEST_PATTERNS_CONFIG = [
    [
        [
            "1 A",
            "2 C",
            "1 -> 2"
        ]
    ],
    [
        [
            "1 A",
            "2 C",
            "3 C",
            "1 -> 2",
            "2 -> 3"
        ],
        [
            "4 A",
            "5 E",
            "6 F",
            "4 -> 5",
            "4 -> 6"
        ]
    ]
]


@pytest.mark.parametrize("custom_patterns", TEST_PATTERNS_CONFIG)
def test_config_parser(custom_patterns):
    pattern = PATTERN_FACTORY.get_full_pattern_graph(custom_patterns)
    for custom_pattern in custom_patterns:
        ref_graph = create_graph_pattern_from_pattern_view(custom_pattern)
        subgraphs = find_subgraphs_matching_expression(ref_graph, pattern)
        assert subgraphs[0]
