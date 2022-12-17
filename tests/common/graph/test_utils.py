import pytest

from nncf.common.graph import NNCFNode
from nncf.common.graph.utils import get_concat_axis
from nncf.common.graph.utils import get_not_matched_scopes
from nncf.parameters import IgnoredScope

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


@pytest.mark.parametrize("scope, ref", [
    ("A", []),
    ("1", ["1"]),
    (["A", "B"], []),
    (["1", "2"], ["1", "2"]),
    ([r"{re}\d"], [r"{re}\d"]),
    ([r"{re}\w"], []),
    (["A", "B", "{re}.*", "1"], ["1"]),
    (IgnoredScope(names=["A", "B"]), []),
    (IgnoredScope(names=["1", "2"]), ["1", "2"]),
    (IgnoredScope(patterns=[r"\d"]), [r"{re}\d"]),
    (IgnoredScope(patterns=[r"\w"]), []),
])
def test_get_not_matched_scopes(scope, ref):
    node_lists = [NNCFNode(1, "A"), NNCFNode(2, "B")]
    not_matched = get_not_matched_scopes(scope, node_lists)
    assert not set(not_matched) - set(ref)
