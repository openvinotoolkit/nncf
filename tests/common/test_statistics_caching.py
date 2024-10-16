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
from collections import deque

import numpy as np
import pytest

import nncf
from nncf.common.tensor_statistics.aggregator import StatisticsSerializer
from nncf.tensor import Tensor
from nncf.tensor.functions import allclose


def _compare_dicts(dict1, dict2):
    """
    Recursively compares two dictionaries.
    Supports comparing numpy arrays and Tensor objects.
    """
    if not isinstance(dict1, dict) or not isinstance(dict2, dict):
        raise ValueError("Both inputs must be dictionaries")

    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                return False
        elif isinstance(val1, Tensor) and isinstance(val2, Tensor):
            if not allclose(val1, val2):
                return False
        # Recursively compare nested dictionaries
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not _compare_dicts(val1, val2):
                return False
        # Direct comparison for other types
        else:
            if val1 != val2:
                return False

    return True


@pytest.fixture
def dummy_statistics():
    """
    Returns a dummy statistics dictionary for testing purposes.
    """
    return {
        "point_A": {"min": 1, "max": 2},
        "point_B": {
            "min_tuple": (1, 2),
            "max_dict": {"tensor_1": [10, 10], "tensor_2": deque([1, 2])},
            "tensor_numpy": Tensor(np.ones(shape=(10, 5, 3))),
        },
    }


def test_dump_and_load_statistics(tmp_path, dummy_statistics):
    """
    Tests that dumped statistics can be loaded and match the original.
    """
    test_file = "test"
    StatisticsSerializer.dump_to_file(dummy_statistics, tmp_path / test_file)
    assert (tmp_path / test_file).exists(), "Dumped file was not created"

    loaded_statistics = StatisticsSerializer.load_from_file(tmp_path / test_file)
    assert _compare_dicts(dummy_statistics, loaded_statistics), "Loaded statistics do not match the original"


def test_load_statistics_from_non_existent_file():
    """
    Tests that attempting to load statistics from a non-existent file raises an error.
    """
    file_path = "non_existent_file"
    with pytest.raises(nncf.ValidationError) as excinfo:
        StatisticsSerializer.load_from_file(file_path)
    assert "File not found" in str(excinfo)
