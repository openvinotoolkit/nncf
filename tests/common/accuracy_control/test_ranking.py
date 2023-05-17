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

from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.accuracy_control.rank_functions import normalized_mse
from nncf.quantization.algorithms.accuracy_control.ranker import get_ranking_subset_indices


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
def test_get_ranking_subset_indices(errors: List[float], subset_size: int, expected_indices: List[int]):
    actual_indices = get_ranking_subset_indices(errors, subset_size)
    assert expected_indices == actual_indices


def _validation_fn_with_error(model, val_dataset) -> float:
    if len(list(val_dataset)) < 3:
        raise RuntimeError
    return 0.1, [0.1]


def _validation_fn(model, val_dataset) -> float:
    return 0.1, [0.1]


class DummyAccuracyControlAlgoBackend:
    @staticmethod
    def prepare_for_inference(model):
        return model


def test_create_logits_ranker():
    algo_backend = DummyAccuracyControlAlgoBackend()
    dataset = Dataset([0, 1, 2])
    evaluator = Evaluator(_validation_fn_with_error, algo_backend)
    evaluator.validate(None, dataset)
    assert not evaluator.is_metric_mode()


def test_create_metric_ranker():
    algo_backend = DummyAccuracyControlAlgoBackend()
    dataset = Dataset([0, 1, 2])
    evaluator = Evaluator(_validation_fn, algo_backend)
    evaluator.validate(None, dataset)
    assert evaluator.is_metric_mode()
