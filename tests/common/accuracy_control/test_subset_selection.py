# Copyright (c) 2024 Intel Corporation
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

from nncf.quantization.algorithms.accuracy_control.subset_selection import get_subset_indices, get_subset_indices_pot_version, select_subset

@pytest.mark.parametrize(
    "errors, subset_size, expected_indices, id",
    [
        ([], 5, [], "empty_list"),
        ([1, 2, 3, 4, 5], 10, [0, 1, 2, 3, 4], "subset_size_larger_than_errors"),
        ([5, 4, 3, 2, 1], 3, [0, 1, 2], "subset_size_smaller_than_errors"),
        ([5, 4, 3, 2, 1], 5, [0, 1, 2, 3, 4], "subset_size_equal_to_errors"),
        ([5, 4, 3, 2, 1], 0, [], "subset_size_zero"),
    ]
)
def test_get_subset_indices(errors, subset_size, expected_indices, id):
    assert(get_subset_indices(errors, subset_size) == expected_indices)
    assert(get_subset_indices_pot_version(errors, subset_size) == expected_indices)

@pytest.mark.parametrize(
    "subset_size, reference_values, approximate_values, expected_indices",
    [
        (5, [], [], []),
        (0, [5, 4, 3, 2, 1], [10, 20, 30, 2], []),
        (10, [1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [0, 1, 2, 3, 4]),
        (3, [5, 4, 3, 2, 1], [10, 20, 30, 2], [0, 1, 2]), 
        (1, [5, 4, 3, 2, 1], [10, 20, 30, 2], [2]),
    ]
)
def test_select_subset(subset_size, reference_values, approximate_values, expected_indices):
    error_fn = lambda x, y: abs(x - y)
    subset_indices = select_subset(subset_size, reference_values, approximate_values, error_fn)
    assert subset_indices == expected_indices









