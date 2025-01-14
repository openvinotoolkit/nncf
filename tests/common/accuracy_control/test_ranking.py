# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import operator
from typing import List
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pytest

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.utils.backend import BackendType
from nncf.quantization.algorithms.accuracy_control.rank_functions import normalized_mse
from nncf.quantization.algorithms.accuracy_control.ranker import GroupToRank
from nncf.quantization.algorithms.accuracy_control.ranker import Ranker
from nncf.quantization.algorithms.accuracy_control.subset_selection import get_subset_indices
from nncf.quantization.algorithms.accuracy_control.subset_selection import get_subset_indices_pot_version
from nncf.quantization.algorithms.accuracy_control.subset_selection import select_subset
from tests.common.accuracy_control.backend import AABackendForTests
from tests.common.quantization.test_quantizer_removal import GRAPHS as AA_GRAPHS_DESCR
from tests.common.quantization.test_quantizer_removal import create_nncf_graph as aa_create_nncf_graph


def create_fp32_tensor_1d(items):
    return np.array(items, dtype=np.float32)


@pytest.mark.parametrize(
    "x_ref,x_approx,expected_nmse",
    [
        # zero_nmse_when_equal
        [
            create_fp32_tensor_1d([1.6784564, 0.415631]),
            create_fp32_tensor_1d([1.6784564, 0.415631]),
            0.0,
        ],
        # trivial
        [
            create_fp32_tensor_1d([2, 1, -1]),
            create_fp32_tensor_1d([-2, 4, 1]),
            4.833333,
        ],
        # not_symmetric
        [
            create_fp32_tensor_1d([-2, 4, 1]),
            create_fp32_tensor_1d([2, 1, -1]),
            1.380952,
        ],
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
    "errors,subset_size,expected_indices",
    [
        # all_different
        [[-0.1, 0.02, 0.2, 0.1, 0.05], 4, [1, 2, 3, 4]],
        # sort_stable
        [[1.0, 2.0, 1.0, 2.0, 3.0, 1.0], 4, [1, 3, 4, 5]],
        # all_equal
        [[0.1, 0.1, 0.1, 0.1], 3, [1, 2, 3]],
        # subset_size_equals_zero
        [[0.1, 0.2, 0.3, 0.4], 0, []],
        # simple
        [[5, 5, 3, 3, 2, 2, 1, 1], 6, [0, 1, 2, 3, 4, 5]],
        # all_negative
        [[-10, -0.1, -5, -5, -0.1, -0.001], 4, [1, 3, 4, 5]],
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
def test_get_subset_indices_pot_version(errors: List[float], subset_size: int, expected_indices: List[int]):
    actual_indices = get_subset_indices_pot_version(errors, subset_size)
    assert expected_indices == actual_indices


@pytest.mark.parametrize(
    "nncf_graph_name,ref_groups",
    [
        (
            "simple_graph",
            [
                GroupToRank(
                    ["fake_quantize_139", "fake_quantize_162", "fake_quantize_119"],
                    ["add_117", "conv2d_161"],
                ),
                GroupToRank(["fake_quantize_153", "fake_quantize_147"], ["conv2d_146"]),
                GroupToRank(["fake_quantize_134", "fake_quantize_128"], ["conv2d_127"]),
            ],
        ),
        (
            "graph_with_shapeof",
            [
                GroupToRank(["fake_quantize_105"], ["interpolate_115"]),
                GroupToRank(["fake_quantize_710", "fake_quantize_93"], ["multiply_99"]),
                GroupToRank(["fake_quantize_82"], ["power_87"]),
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


def _validation_fn(model, dataset, indices):
    return (0.1, [0.1])


def collect_logits(model, dataset, indices):
    sample_logits = [[create_fp32_tensor_1d([-2, 4, 1])]]
    return sample_logits


@pytest.fixture
def evaluator_and_ranker():
    evaluator = Mock()
    evaluator.validate_prepared_model = _validation_fn
    evaluator.collect_values_for_each_item_using_prepared_model = collect_logits
    ranker = Ranker(1, tuple(), AABackendForTests, evaluator=evaluator, ranking_fn=None)
    return evaluator, ranker


@pytest.mark.parametrize(
    "backend, metric_mode, expected_result",
    [
        (BackendType.OPENVINO, True, operator.sub),
        (BackendType.TENSORFLOW, True, operator.sub),
        (BackendType.OPENVINO, False, callable),
    ],
)
def test_create_ranking_fn(backend, metric_mode, expected_result, evaluator_and_ranker):
    evaluator, ranker = evaluator_and_ranker
    evaluator.is_metric_mode.return_value = metric_mode
    ranking_fn = ranker._create_ranking_fn(backend)
    assert ranking_fn == expected_result or expected_result(ranking_fn)


def test_create_ranking_fn_error(evaluator_and_ranker):
    evaluator, ranker = evaluator_and_ranker
    evaluator.is_metric_mode.return_value = False
    with pytest.raises(nncf.UnsupportedBackendError):
        ranker._create_ranking_fn(BackendType.TENSORFLOW)


@pytest.fixture
def ranking_subset_indices_and_ref_values():
    ranking_subset_indices = [0]
    reference_values_for_each_item = [[create_fp32_tensor_1d([2, 1, -1])]]
    return ranking_subset_indices, reference_values_for_each_item


@pytest.fixture
def quantized_model_and_graph():
    quantized_model_graph = aa_create_nncf_graph(AA_GRAPHS_DESCR["simple_graph"])
    quantized_model = Mock()
    return quantized_model, quantized_model_graph


@pytest.mark.parametrize("is_metric_mode, expected_score", [(True, 0.1), (False, 4.833333)])
def test_calculate_ranking_score(
    evaluator_and_ranker,
    ranking_subset_indices_and_ref_values,
    is_metric_mode,
    expected_score,
):
    evaluator, ranker = evaluator_and_ranker
    ranking_subset_indices, reference_values_for_each_item = ranking_subset_indices_and_ref_values
    evaluator.is_metric_mode.return_value = is_metric_mode
    ranker._ranking_fn = ranker._create_ranking_fn(BackendType.OPENVINO)
    prepared_model = Mock()
    assert np.allclose(
        expected_score,
        ranker._calculate_ranking_score(prepared_model, ranking_subset_indices, reference_values_for_each_item),
    )


def test_sequential_calculation_ranking_score(
    evaluator_and_ranker, ranking_subset_indices_and_ref_values, mocker, quantized_model_and_graph
):
    quantized_model, quantized_model_graph = quantized_model_and_graph

    evaluator, ranker = evaluator_and_ranker

    # mock prepare_model and revert_operations_to_floating_point to simply return the passed model
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=quantized_model
    )
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.ranker.revert_operations_to_floating_point_precision",
        return_value=quantized_model,
    )

    ranking_subset_indices, reference_values_for_each_item = ranking_subset_indices_and_ref_values

    evaluator.is_metric_mode.return_value = False
    ranker._ranking_fn = ranker._create_ranking_fn(BackendType.OPENVINO)

    groups_to_rank = ranker.find_groups_of_quantizers_to_rank(quantized_model_graph)
    scores: List[float] = ranker._sequential_calculation_ranking_score(
        quantized_model,
        quantized_model_graph,
        groups_to_rank,
        ranking_subset_indices,
        reference_values_for_each_item,
    )
    assert len(scores) == len(groups_to_rank)
    for s in scores:
        assert np.allclose(s, 4.833333)


def test_rank_groups_of_quantizers_score_all_same(
    evaluator_and_ranker, ranking_subset_indices_and_ref_values, mocker, quantized_model_and_graph
):
    quantized_model, quantized_model_graph = quantized_model_and_graph

    evaluator, ranker = evaluator_and_ranker
    ranking_subset_indices, reference_values_for_each_item = ranking_subset_indices_and_ref_values
    approximate_values_for_each_item = [[create_fp32_tensor_1d([2, 1, -1])]]
    evaluator.is_metric_mode.return_value = False
    ranker._ranking_fn = ranker._create_ranking_fn(BackendType.OPENVINO)
    groups_to_rank = ranker.find_groups_of_quantizers_to_rank(quantized_model_graph)

    mock_subset_selection = mocker.patch("nncf.quantization.algorithms.accuracy_control.subset_selection.select_subset")
    mock_subset_selection.return_value = ranking_subset_indices

    # mock prepare_model and revert_operations_to_floating_point to simply return the passed model
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=quantized_model
    )
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.ranker.revert_operations_to_floating_point_precision",
        return_value=quantized_model,
    )

    ranked_groups: GroupToRank = ranker.rank_groups_of_quantizers(
        groups_to_rank,
        quantized_model,
        quantized_model_graph,
        reference_values_for_each_item,
        approximate_values_for_each_item,
    )
    assert len(ranked_groups) == len(groups_to_rank)


def test_rank_groups_of_quantizers_score_different(
    evaluator_and_ranker, ranking_subset_indices_and_ref_values, mocker, quantized_model_and_graph
):
    quantized_model, quantized_model_graph = quantized_model_and_graph
    evaluator, ranker = evaluator_and_ranker
    ranking_subset_indices, reference_values_for_each_item = ranking_subset_indices_and_ref_values
    approximate_values_for_each_item = [[create_fp32_tensor_1d([2, 1, -1])]]
    evaluator.is_metric_mode.return_value = False
    ranker._ranking_fn = ranker._create_ranking_fn(BackendType.OPENVINO)
    groups_to_rank = ranker.find_groups_of_quantizers_to_rank(quantized_model_graph)

    mock_subset_selection = mocker.patch("nncf.quantization.algorithms.accuracy_control.subset_selection.select_subset")
    mock_subset_selection.return_value = ranking_subset_indices

    # mock prepare_model and revert_operations_to_floating_point to simply return the passed model
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=quantized_model
    )
    mocker.patch(
        "nncf.quantization.algorithms.accuracy_control.ranker.revert_operations_to_floating_point_precision",
        return_value=quantized_model,
    )

    mock_scores = [1.0, 2.0, 3.0]
    with patch.object(ranker, "_sequential_calculation_ranking_score", return_value=mock_scores):
        ranked_groups: GroupToRank = ranker.rank_groups_of_quantizers(
            groups_to_rank,
            quantized_model,
            quantized_model_graph,
            reference_values_for_each_item,
            approximate_values_for_each_item,
        )
        assert ranked_groups == groups_to_rank

    mock_scores = [3.0, 2.0, 1.0]
    with patch.object(ranker, "_sequential_calculation_ranking_score", return_value=mock_scores):
        ranked_groups: GroupToRank = ranker.rank_groups_of_quantizers(
            groups_to_rank,
            quantized_model,
            quantized_model_graph,
            reference_values_for_each_item,
            approximate_values_for_each_item,
        )
        assert ranked_groups == groups_to_rank[::-1]


@pytest.mark.parametrize(
    "subset_size, reference_values, approximate_values, expected_indices",
    [
        (5, [], [], []),
        (0, [5, 4, 3, 2, 1], [10, 20, 30, 2], []),
        (10, [1, 2, 3, 4, 5], [1, 1, 1, 1, 1], [0, 1, 2, 3, 4]),
        (3, [5, 4, 3, 2, 1], [10, 20, 30, 2], [0, 1, 2]),
        (1, [5, 4, 3, 2, 1], [10, 20, 30, 2], [2]),
    ],
)
def test_select_subset(subset_size, reference_values, approximate_values, expected_indices):
    error_fn = lambda x, y: abs(x - y)
    subset_indices = select_subset(subset_size, reference_values, approximate_values, error_fn)
    assert subset_indices == expected_indices
