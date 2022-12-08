import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.operator_metatypes import InputNoopMetatype
from nncf.common.graph.utils import check_scope_names_match_graph
from nncf.common.graph.utils import get_concat_axis
from nncf.config import NNCFConfig


TEST_CASES = [
    ([(1, 1), (1, 1)], [(2, 1)], [0]),
    ([(None, 1, 1, 5)], [(None, 1, 1, 7)], [3, -1]),
    ([(None, 1, 1, 5), (None, 1, 1, 5)], [(None, 1, 1, 10)], [3, -1]),
    ([(1, 1, None), (1, 1, None)], [(1, 1, None)], [2, -1]),
    ([(1, 1, 32, 1), (1, 1, 32, 1)], [(1, 1, 64, 1)], [2, -1]),
    ([(1, 1, 5), (1, 1, 5)], [(1, 1, 5)], [-1]),
]


@pytest.mark.parametrize('input_shape,output_shape,possible_axes', TEST_CASES)
def test_get_concat_axis(input_shape, output_shape, possible_axes):
    axis = get_concat_axis(input_shape, output_shape)
    assert axis in possible_axes


@pytest.mark.parametrize("update_config, should_fail", [
    ({}, False),
    ({"ignored_scopes":  "12"}, True),
    ({"ignored_scopes": ["{re}1"]}, False),
    ({"ignored_scopes": ["{re}A"]}, True),
    ({"ignored_scopes": ["A", "B"]}, True),
    ({"target_scopes":  ["A", "B"]}, True),
    ({"ignored_scopes": ["1", "2"]}, False),
    ({"target_scopes":  ["1", "2"]}, False),
    ({"compression": {"algorithm": "quantization", "ignored_scopes": ["A", "B"]}}, True),
    ({"compression": {"algorithm": "quantization", "target_scopes":  ["A", "B"]}}, True),
    ({"compression": {"algorithm": "quantization", "ignored_scopes": ["1", "2"]}}, False),
    ({"compression": {"algorithm": "quantization", "target_scopes":  ["1", "2"]}}, False),
    ({  "ignored_scopes":  "12",
        "compression": [
            {"algorithm": "quantization", "ignored_scopes": ["12"], "target_scopes": ["A", "B"]},
            {"algorithm": "filter_pruning", "ignored_scopes": ["A", "B"], "target_scopes": ["12"]},
        ]
    }, True),
])
def test_check_scope_names_match_graph(update_config, should_fail):
    graph = NNCFGraph()
    graph.add_nncf_node("1", "conv2d", InputNoopMetatype)
    graph.add_nncf_node("2", "conv2d", InputNoopMetatype)

    config = NNCFConfig()
    config.update(update_config)

    if should_fail:
        with pytest.raises(RuntimeError):
            check_scope_names_match_graph(config, graph)
    else:
        check_scope_names_match_graph(config, graph)
