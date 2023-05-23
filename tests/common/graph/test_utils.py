# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from nncf.common.graph.utils import get_concat_axis

TEST_CASES = [
    ([(1, 1), (1, 1)], [(2, 1)], [0]),
    ([(None, 1, 1, 5)], [(None, 1, 1, 7)], [3, -1]),
    ([(None, 1, 1, 5), (None, 1, 1, 5)], [(None, 1, 1, 10)], [3, -1]),
    ([(1, 1, None), (1, 1, None)], [(1, 1, None)], [2, -1]),
    ([(1, 1, 32, 1), (1, 1, 32, 1)], [(1, 1, 64, 1)], [2, -1]),
    ([(1, 1, 5), (1, 1, 5)], [(1, 1, 5)], [-1]),
]


@pytest.mark.parametrize("input_shape,output_shape,possible_axes", TEST_CASES)
def test_get_concat_axis(input_shape, output_shape, possible_axes):
    axis = get_concat_axis(input_shape, output_shape)
    assert axis in possible_axes
