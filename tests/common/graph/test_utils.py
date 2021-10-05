import pytest

from nncf.common.graph.utils import get_concat_axis


TEST_CASES = [
    ([(None, 1, 1, 5)], [(None, 1, 1, 5)], False, [3, -1]),
    ([(None, 1, 1, 5), (None, 1, 1, 5)], [(None, 1, 1, 10)], False, [3, -1]),
    ([(1, 1, None), (1, 1, None)], [(1, 1, None)], False, [2, -1]),
    ([(1, 1, 32, 1), (1, 1, 32, 1)], [(1, 1, 64, 1)], False, [2, -1]),
    ([(1, 1, 5), (1, 1, 5)], [(1, 1, 5)], True, None),
]


@pytest.mark.parametrize('input_shape,output_shape,raise_error,possible_axes', TEST_CASES)
def test_get_concat_axis(input_shape, output_shape, raise_error, possible_axes):
    if not raise_error:
        assert get_concat_axis(input_shape, output_shape) in possible_axes
    else:
        with pytest.raises(RuntimeError):
            _ = get_concat_axis(input_shape, output_shape)
