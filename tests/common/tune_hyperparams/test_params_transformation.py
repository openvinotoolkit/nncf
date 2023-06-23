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

from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, List

import pytest

from nncf.quantization.algorithms.tune_hyperparams.params_transformation import ParamsTransformation
from nncf.quantization.algorithms.tune_hyperparams.params_transformation import create_params_transformation

# =========================================================
# TEST: Create params transformation
# =========================================================


@pytest.mark.parametrize(
    "search_space,expected_suppositions",
    [
        # plain
        (
            # search_space
            {"a": [1, 2, 3]},
            # expected_suppositions
            {
                "a": [ParamsTransformation({"a": 1}), ParamsTransformation({"a": 2}), ParamsTransformation({"a": 3})],
            },
        ),
        # nested_dict
        (
            # search_space
            {"a": {"b": [1, 2]}},
            # expected_suppositions
            {
                "a": [ParamsTransformation({"a": {"b": 1}}), ParamsTransformation({"a": {"b": 2}})],
            },
        ),
        # plain_and_nested_dict
        (
            # search_space
            {"x": [0, 1], "y": {"a": [10, 20]}},
            # expected_suppositions
            {
                "x": [ParamsTransformation({"x": 0}), ParamsTransformation({"x": 1})],
                "y": [ParamsTransformation({"y": {"a": 10}}), ParamsTransformation({"y": {"a": 20}})],
            },
        ),
        # complex
        (
            # search_space
            {
                "x": [0],
                "y": {
                    "a": [0, 1],
                    "b": [2, 3],
                    "d": {
                        "e": [-1],
                        "f": {
                            "x": [1],
                            "y": [-1, 2],
                        },
                    },
                },
            },
            # expected_suppositions
            {
                "x": [
                    ParamsTransformation({"x": 0}),
                ],
                "y": [
                    ParamsTransformation({"y": {"a": 0}}),
                    ParamsTransformation({"y": {"a": 1}}),
                    ParamsTransformation({"y": {"b": 2}}),
                    ParamsTransformation({"y": {"b": 3}}),
                    ParamsTransformation({"y": {"d": {"e": -1}}}),
                    ParamsTransformation({"y": {"d": {"f": {"x": 1}}}}),
                    ParamsTransformation({"y": {"d": {"f": {"y": -1}}}}),
                    ParamsTransformation({"y": {"d": {"f": {"y": 2}}}}),
                ],
            },
        ),
    ],
    ids=[
        "plain",
        "nested_dict",
        "plain_and_nested_dict",
        "complex",
    ],
)
def test_create_params_transformation(search_space, expected_suppositions):
    actual_suppositions = create_params_transformation(search_space)

    for param_name, expected_param_value_suppositions in expected_suppositions.items():
        actual_param_value_suppositions = actual_suppositions[param_name]
        for expected, actual in zip(expected_param_value_suppositions, actual_param_value_suppositions):
            assert expected._changes == actual._changes


# =========================================================
# TEST: Apply params transformation
# =========================================================


@dataclass
class ParamsB:
    y: int = None
    z: str = None


@dataclass
class ParamsA:
    x: bool = None
    b: ParamsB = field(default_factory=ParamsB)


@pytest.mark.parametrize(
    "params,params_transformation,expected_params",
    [
        ({"a": None}, ParamsTransformation({"a": -1}), {"a": -1}),
        ({"a": ParamsA()}, ParamsTransformation({"a": ParamsA(x=True)}), {"a": ParamsA(x=True)}),
        ({"a": ParamsA()}, ParamsTransformation({"a": {"x": True}}), {"a": ParamsA(x=True)}),
        (
            {"a": ParamsA()},
            ParamsTransformation({"a": {"x": False, "b": {"y": -1}}}),
            {"a": ParamsA(x=False, b=ParamsB(y=-1))},
        ),
    ],
)
def test_apply_params_transformation(
    params: Dict[str, Any], params_transformation: ParamsTransformation, expected_params: Dict[str, Any]
):
    actual_params = params_transformation.apply(params)
    assert expected_params == actual_params
