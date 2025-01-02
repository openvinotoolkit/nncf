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

import copy
from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List, Tuple

import pytest

from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import apply_combination
from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import create_combinations
from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import find_best_combination

CombinationKey = Tuple[int, ...]
Combination = Dict[str, Any]


# =========================================================
# TEST: Create combinations
# =========================================================


@pytest.mark.parametrize(
    "param_settings,expected_combinations",
    [
        # one_parameter
        (
            # param_settings
            {"x": [0, 1, 2]},
            # expected_combinations
            {
                (None,): {},
                (0,): {"x": 0},
                (1,): {"x": 1},
                (2,): {"x": 2},
            },
        ),
        # two_parameters
        (
            # param_settings
            {"x": [0, 1, 2], "y": [True, False]},
            # expected_combinations
            {
                (None, None): {},
                (None, 0): {"y": True},
                (None, 1): {"y": False},
                (0, None): {"x": 0},
                (0, 0): {"x": 0, "y": True},
                (0, 1): {"x": 0, "y": False},
                (1, None): {"x": 1},
                (1, 0): {"x": 1, "y": True},
                (1, 1): {"x": 1, "y": False},
                (2, None): {"x": 2},
                (2, 0): {"x": 2, "y": True},
                (2, 1): {"x": 2, "y": False},
            },
        ),
        # three_parameters
        (
            # param_settings
            {"x": [0, 1], "y": [True, False], "z": ["a", "b"]},
            # expected_combinations
            {
                (None, None, None): {},
                (None, None, 0): {"z": "a"},
                (None, None, 1): {"z": "b"},
                (None, 0, None): {"y": True},
                (None, 0, 0): {"y": True, "z": "a"},
                (None, 0, 1): {"y": True, "z": "b"},
                (None, 1, None): {"y": False},
                (None, 1, 0): {"y": False, "z": "a"},
                (None, 1, 1): {"y": False, "z": "b"},
                (0, None, None): {"x": 0},
                (0, None, 0): {"x": 0, "z": "a"},
                (0, None, 1): {"x": 0, "z": "b"},
                (0, 0, None): {"x": 0, "y": True},
                (0, 0, 0): {"x": 0, "y": True, "z": "a"},
                (0, 0, 1): {"x": 0, "y": True, "z": "b"},
                (0, 1, None): {"x": 0, "y": False},
                (0, 1, 0): {"x": 0, "y": False, "z": "a"},
                (0, 1, 1): {"x": 0, "y": False, "z": "b"},
                (1, None, None): {"x": 1},
                (1, None, 0): {"x": 1, "z": "a"},
                (1, None, 1): {"x": 1, "z": "b"},
                (1, 0, None): {"x": 1, "y": True},
                (1, 0, 0): {"x": 1, "y": True, "z": "a"},
                (1, 0, 1): {"x": 1, "y": True, "z": "b"},
                (1, 1, None): {"x": 1, "y": False},
                (1, 1, 0): {"x": 1, "y": False, "z": "a"},
                (1, 1, 1): {"x": 1, "y": False, "z": "b"},
            },
        ),
    ],
    ids=[
        "one_parameter",
        "two_parameters",
        "three_parameters",
    ],
)
def test_create_combinations(
    param_settings: Dict[str, List[Any]], expected_combinations: Dict[CombinationKey, Combination]
):
    actual_combinations = create_combinations(param_settings)
    assert expected_combinations == actual_combinations


# =========================================================
# TEST: Apply combination
# =========================================================


@dataclass
class ParamsA:
    x: int = None
    y: str = None


@dataclass
class ParamsB:
    x: bool = None
    a: ParamsA = field(default_factory=ParamsA)


@dataclass
class ParamsC:
    x: float = None
    b: ParamsB = field(default_factory=ParamsB)


@pytest.mark.parametrize(
    "init_params,combination,expected_params",
    [
        # change_one_parameter
        (
            # init_params
            {"x": 0, "y": True},
            # combination
            {"x": -1},
            # expected_params
            {"x": -1, "y": True},
        ),
        # change_two_parameters
        (
            # init_params
            {"x": 0, "y": True, "z": "a"},
            # combination
            {"x": -1, "y": False},
            # expected_params
            {"x": -1, "y": False, "z": "a"},
        ),
        # change_all_parameters
        (
            # init_params
            {"x": 0, "y": True, "z": "a"},
            # combination
            {"x": -1, "y": False, "z": "b"},
            # expected_params
            {"x": -1, "y": False, "z": "b"},
        ),
        # change_one_subparameter_depth_1
        (
            # init_params
            {"x": 0, "p": ParamsA(), "y": True},
            # combination
            {"p:x": -1},
            # expected_params
            {"x": 0, "p": ParamsA(x=-1), "y": True},
        ),
        # change_one_parameter_and_one_subparameter_depth_1
        (
            # init_params
            {"x": 0, "p": ParamsA(), "y": True},
            # combination
            {"x": -1, "p:y": "a"},
            # expected_params
            {"x": -1, "p": ParamsA(y="a"), "y": True},
        ),
        # change_all_subparameters_depth_1
        (
            # init_params
            {"x": 0, "p": ParamsA(), "y": True},
            # combination
            {"p:x": -1, "p:y": "a"},
            # expected_params
            {"x": 0, "p": ParamsA(x=-1, y="a"), "y": True},
        ),
        # change_one_subparameter_depth_3
        (
            # init_params
            {"x": 0, "p": ParamsC()},
            # combination
            {"p:b:a:x": -1},
            # expected_params
            {"x": 0, "p": ParamsC(b=ParamsB(a=ParamsA(x=-1)))},
        ),
        # change_one_subparameter_depth_2
        (
            # init_params
            {"x": 0, "p": ParamsC()},
            # combination
            {"p:b:a": ParamsA(x=-1, y="a")},
            # expected_params
            {"x": 0, "p": ParamsC(b=ParamsB(a=ParamsA(x=-1, y="a")))},
        ),
        # change_one_parameter_and_one_subparameter_depth_3
        (
            # init_params
            {"x": 0, "p": ParamsC(), "y": True},
            # combination
            {"p:b:a:x": -1, "y": False},
            # expected_params
            {"x": 0, "p": ParamsC(b=ParamsB(a=ParamsA(x=-1))), "y": False},
        ),
    ],
    ids=[
        "change_one_parameter",
        "change_two_parameters",
        "change_all_parameters",
        "change_one_subparameter_depth_1",
        "change_one_parameter_and_one_subparameter_depth_1",
        "change_all_subparameters_depth_1",
        "change_one_subparameter_depth_3",
        "change_one_subparameter_depth_2",
        "change_one_parameter_and_one_subparameter_depth_3",
    ],
)
def test_apply_combination(init_params: Dict[str, Any], combination: Combination, expected_params: Dict[str, Any]):
    init_params_copy = copy.deepcopy(init_params)
    actual_params = apply_combination(init_params, combination)

    # Check that `init_params` was not changed
    assert init_params_copy == init_params

    assert expected_params == actual_params


# =========================================================
# TEST: Find best combination
# =========================================================


@pytest.mark.parametrize(
    "combinations,scores,param_settings,expected_combination_key",
    [
        # one_parameter-scores_different
        (
            # combinations
            {
                (None,): {},
                (0,): {"x": 0},
                (1,): {"x": 1},
            },
            # scores
            {
                (None,): 0.1,
                (0,): 0.3,
                (1,): 0.2,
            },
            # param_settings
            {
                "x": [0, 1],
            },
            # expected_combination_key
            (0,),
        ),
        # one_parameter-scores_equal
        (
            # combinations
            {
                (None,): {},
                (0,): {"x": 0},
                (1,): {"x": 1},
            },
            # scores
            {
                (None,): 0.1,
                (0,): 0.1,
                (1,): 0.1,
            },
            # param_settings
            {
                "x": [0, 1],
            },
            # expected_combination_key
            (None,),
        ),
        # two_parameter-scores_different
        (
            # combinations
            {
                (None, None): {},
                (None, 0): {"y": True},
                (None, 1): {"y": False},
                (0, None): {"x": 0},
                (0, 0): {"x": 0, "y": True},
                (0, 1): {"x": 0, "y": False},
                (1, None): {"x": 1},
                (1, 0): {"x": 1, "y": True},
                (1, 1): {"x": 1, "y": False},
                (2, None): {"x": 2},
                (2, 0): {"x": 2, "y": True},
                (2, 1): {"x": 2, "y": False},
            },
            # scores
            {
                (None, None): 0.1,
                (0, None): 0.2,
                (1, None): 0.4,
                (2, None): 0.1,
                (1, 0): 0.6,
                (1, 1): 0.7,
            },
            # param_settings
            {
                "x": [0, 1, 2],
                "y": [True, False],
            },
            # expected_combination_key
            (1, 1),
        ),
        # two_parameters-no_best_value_for_second_parameter
        (
            # combinations
            {
                (None, None): {},
                (None, 0): {"y": True},
                (None, 1): {"y": False},
                (0, None): {"x": 0},
                (0, 0): {"x": 0, "y": True},
                (0, 1): {"x": 0, "y": False},
                (1, None): {"x": 1},
                (1, 0): {"x": 1, "y": True},
                (1, 1): {"x": 1, "y": False},
                (2, None): {"x": 2},
                (2, 0): {"x": 2, "y": True},
                (2, 1): {"x": 2, "y": False},
            },
            # scores
            {
                (None, None): 0.1,
                (0, None): 0.5,
                (1, None): 0.2,
                (2, None): 0.3,
                (0, 0): 0.3,
                (0, 1): 0.4,
            },
            # param_settings
            {
                "x": [0, 1, 2],
                "y": [True, False],
            },
            # expected_combination_key
            (0, None),
        ),
        # three_parameters-no_best_value_for_second_parameter
        (
            # combinations
            {
                (None, None, None): {},
                (None, None, 0): {"z": "a"},
                (None, None, 1): {"z": "b"},
                (None, 0, None): {"y": True},
                (None, 0, 0): {"y": True, "z": "a"},
                (None, 0, 1): {"y": True, "z": "b"},
                (None, 1, None): {"y": False},
                (None, 1, 0): {"y": False, "z": "a"},
                (None, 1, 1): {"y": False, "z": "b"},
                (0, None, None): {"x": 0},
                (0, None, 0): {"x": 0, "z": "a"},
                (0, None, 1): {"x": 0, "z": "b"},
                (0, 0, None): {"x": 0, "y": True},
                (0, 0, 0): {"x": 0, "y": True, "z": "a"},
                (0, 0, 1): {"x": 0, "y": True, "z": "b"},
                (0, 1, None): {"x": 0, "y": False},
                (0, 1, 0): {"x": 0, "y": False, "z": "a"},
                (0, 1, 1): {"x": 0, "y": False, "z": "b"},
                (1, None, None): {"x": 1},
                (1, None, 0): {"x": 1, "z": "a"},
                (1, None, 1): {"x": 1, "z": "b"},
                (1, 0, None): {"x": 1, "y": True},
                (1, 0, 0): {"x": 1, "y": True, "z": "a"},
                (1, 0, 1): {"x": 1, "y": True, "z": "b"},
                (1, 1, None): {"x": 1, "y": False},
                (1, 1, 0): {"x": 1, "y": False, "z": "a"},
                (1, 1, 1): {"x": 1, "y": False, "z": "b"},
            },
            # scores
            {
                (None, None, None): 0.5,
                (0, None, None): 0.4,
                (1, None, None): 0.3,
                (None, 0, None): 0.6,
                (None, 1, None): 0.7,
                (None, 1, 0): 0.6,
                (None, 1, 1): 0.5,
            },
            # param_settings
            {
                "x": [0, 1],
                "y": [2, 3],
                "z": [4, 5],
            },
            # expected_combination_key
            (None, 1, None),
        ),
    ],
    ids=[
        "one_parameter-scores_different",
        "one_parameter-scores_equal",
        "two_parameters-scores_different",
        "two_parameters-no_best_value_for_second_parameter",
        "three_parameters-no_best_value_for_first_and_third_parameter",
    ],
)
def test_find_best_combination(
    combinations: Dict[CombinationKey, Combination],
    scores: Dict[CombinationKey, float],
    param_settings: Dict[str, List[Any]],
    expected_combination_key: CombinationKey,
):
    combination_score_func = lambda x: scores[x]
    actual_combination_key = find_best_combination(combinations, combination_score_func, param_settings)
    assert expected_combination_key == actual_combination_key


@pytest.mark.parametrize(
    "params, number_considered_combinations",
    [
        (
            {
                "param_1": [0, 1, 2],
                "param_2": [True, False],
            },
            # Explanation:
            # 2 combinations for default values of each parameter,
            # 5 combinations for the sum of values of each parameter.
            2 + 5,
        ),
        (
            {
                "param_1": [0, 1],
                "param_2": [2, 3, 8],
                "param_3": [4, 5],
            },
            # Explanation:
            # 3 combinations for default values of each parameter,
            # 6 combinations for the sum of values of each parameter.
            3 + 7,
        ),
    ],
)
def test_number_of_combinations_considered(params, number_considered_combinations):
    score_function_call_count = 0

    def score_f(key):
        nonlocal score_function_call_count
        score_function_call_count += 1
        return 1

    combinations = create_combinations(params)
    find_best_combination(combinations, score_f, params)
    assert number_considered_combinations == score_function_call_count, (
        f"Expected {number_considered_combinations} combinations to be considered, "
        f"but got {score_function_call_count}."
    )
