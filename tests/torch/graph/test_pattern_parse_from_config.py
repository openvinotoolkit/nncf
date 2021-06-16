from typing import List
import pytest

from nncf.torch.graph.patterns import PATTERN_FACTORY
from nncf.common.graph.graph_matching import find_subgraphs_matching_expression
from nncf.torch.graph.patterns import create_graph_pattern_from_pattern_view


def create_ref_graph(pattern_view: List[str]):
    graph = create_graph_pattern_from_pattern_view(pattern_view).graph
    for node in graph.nodes:
        graph.nodes[node]["type"] = graph.nodes[node]["type"][0]
    return graph


TEST_PATTERNS_CONFIG = [
    [
        [
            "1 type=[A, B]",
            "2 type=[C]",
            "1 -> 2"
        ]
    ],
    [
        [
            "1 type=[A, B]",
            "2 type=[C]",
            "3 type=[C]",
            "1 -> 2",
            "2 -> 3"
        ],
        [
            "4 type=[A]",
            "5 type=[E]",
            "6 type=[F]",
            "4 -> 5",
            "4 -> 6"
        ]
    ]
]


@pytest.mark.parametrize("custom_patterns", TEST_PATTERNS_CONFIG)
def test_config_parser(custom_patterns):
    pattern = PATTERN_FACTORY.get_full_pattern_graph(custom_patterns)
    for custom_pattern in custom_patterns:
        ref_graph = create_ref_graph(custom_pattern)
        subgraphs = find_subgraphs_matching_expression(ref_graph, pattern)
        assert subgraphs[0]
