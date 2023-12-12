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


from typing import List

import numpy as np
import pytest

from nncf.common.graph.graph import NNCFGraph
from nncf.quantization.algorithms.accuracy_control.rank_functions import normalized_mse
from nncf.quantization.algorithms.accuracy_control.ranker import GroupToRank
from nncf.quantization.algorithms.accuracy_control.ranker import Ranker
from nncf.quantization.algorithms.accuracy_control.subset_selection import get_subset_indices
from tests.common.accuracy_control.backend import AABackendForTests
from tests.common.quantization.test_quantizer_removal import GRAPHS as AA_GRAPHS_DESCR
from tests.common.quantization.test_quantizer_removal import create_nncf_graph as aa_create_nncf_graph


def create_fp32_tensor_1d(items):
    return np.array(items, dtype=np.float32)


@pytest.mark.parametrize(
    "x_ref,x_approx,expected_nmse",
    [
        # zero_nmse_when_equal
        [create_fp32_tensor_1d([1.6784564, 0.415631]), create_fp32_tensor_1d([1.6784564, 0.415631]), 0.0],
        # trivial
        [create_fp32_tensor_1d([2, 1, -1]), create_fp32_tensor_1d([-2, 4, 1]), 4.833333],
        # not_symmetric
        [create_fp32_tensor_1d([-2, 4, 1]), create_fp32_tensor_1d([2, 1, -1]), 1.380952],
    ],
    ids=[
        "zero_nmse_when_equal",
        "trivial",
        "not_symmetric",
    ],
)
def test_normalized_mse(x_ref: np.ndarray, x_approx: np.ndarray, expected_nmse: float):
    actual_nmse = normalized_mse([x_ref], [x_approx])
    assert np.allclose(expected_nmse, actual_nmse)


@pytest.mark.parametrize(
    "errors,subset_size,expected_indices",
    [
        # all_different
        [[-0.1, 0.02, 0.2, 0.1, 0.05], 4, [1, 2, 3, 4]],
        # sort_stable
        [[1.0, 2.0, 1.0, 2.0, 3.0, 1.0], 4, [0, 1, 3, 4]],
        # all_equal
        [[0.1, 0.1, 0.1, 0.1], 3, [0, 1, 2]],
        # subset_size_equals_zero
        [[0.1, 0.2, 0.3, 0.4], 0, []],
        # simple
        [[5, 5, 3, 3, 2, 2, 1, 1], 6, [0, 1, 2, 3, 4, 5]],
        # all_negative
        [[-10, -0.1, -5, -5, -0.1, -0.001], 4, [1, 2, 4, 5]],
        # subset_size_equals_num_errors
        [[0, 1, 2, 3, 4, 5], 6, [0, 1, 2, 3, 4, 5]],
        # subset_size_greater_than_num_errors
        [[0, -1, -2, -3, -4, -5], 10000, [0, 1, 2, 3, 4, 5]],
    ],
    ids=[
        "all_different",
        "sort_stable",
        "all_equal",
        "subset_size_equals_zero",
        "simple",
        "all_negative",
        "subset_size_equals_num_errors",
        "subset_size_greater_than_num_errors",
    ],
)
def test_get_subset_indices(errors: List[float], subset_size: int, expected_indices: List[int]):
    actual_indices = get_subset_indices(errors, subset_size)
    assert expected_indices == actual_indices


@pytest.mark.parametrize(
    "nncf_graph_name,ref_groups",
    [
        (
            "simple_graph",
            [
                GroupToRank(["quantizer_139", "quantizer_162", "quantizer_119"], ["add_117", "conv2d_161"]),
                GroupToRank(["quantizer_153", "quantizer_147"], ["conv2d_146"]),
                GroupToRank(["quantizer_134", "quantizer_128"], ["conv2d_127"]),
            ],
        ),
        (
            "graph_with_shapeof",
            [
                GroupToRank(["quantizer_105"], ["interpolate_115"]),
                GroupToRank(["quantizer_710", "quantizer_93"], ["multiply_99"]),
                GroupToRank(["quantizer_82"], ["power_87"]),
            ],
        ),
    ],
)
def test_find_groups_of_quantizers_to_rank(nncf_graph_name: NNCFGraph, ref_groups: List[GroupToRank]):
    ranker = Ranker(1, tuple(), AABackendForTests, None)
    nncf_graph = aa_create_nncf_graph(AA_GRAPHS_DESCR[nncf_graph_name])
    ret_val = ranker.find_groups_of_quantizers_to_rank(nncf_graph)
    assert len(ret_val) == len(ref_groups)
    # Can zip as qauantizers are topologically sorted
    for actual_group, ref_group in zip(ret_val, ref_groups):
        for attr in ["quantizers", "operations"]:
            acutal_attr_value = getattr(actual_group, attr)
            ref_attr_value = getattr(ref_group, attr)

            assert len(acutal_attr_value) == len(ref_attr_value)
            actual_node_names = [n.node_name for n in acutal_attr_value]
            for ref_node_name in ref_attr_value:
                assert ref_node_name in actual_node_names
