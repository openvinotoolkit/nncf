# Copyright (c) 2026 Intel Corporation
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
from typing import Union

import pytest

from nncf.tensor import Tensor
from nncf.tensor.functions.dispatcher import _get_arg_type
from nncf.tensor.functions.dispatcher import _get_register_types
from nncf.tensor.functions.dispatcher import _unwrap_tensors
from nncf.tensor.functions.dispatcher import _wrap_output


@pytest.mark.parametrize(
    "data, ref",
    (
        (1, 1),
        (Tensor(1), 1),
        ([1, 1], [1, 1]),
        ([Tensor(1), Tensor(1)], [1, 1]),
        ((1, 1), (1, 1)),
        ((Tensor(1), Tensor(1)), (1, 1)),
        ({"a": 1, "b": 1}, {"a": 1, "b": 1}),
        ({"a": Tensor(1), "b": [Tensor(1)], "c": (Tensor(1), 1)}, {"a": 1, "b": [1], "c": (1, 1)}),
    ),
)
def test_unwrap_tensors(data, ref):
    assert _unwrap_tensors(data) == ref


@pytest.mark.parametrize(
    "data, ret_ann, ref",
    (
        (1, int, 1),
        (1, Tensor, Tensor(1)),
        ([1, 1], list[int], [1, 1]),
        ([1, 1], list[Tensor], [Tensor(1), Tensor(1)]),
        ((1, 1), tuple[Tensor, int], (Tensor(1), 1)),
    ),
)
def test_wrap_output(data, ret_ann, ref):
    assert _wrap_output(data, ret_ann) == ref


@pytest.mark.parametrize(
    "data, ref",
    (
        (1, int),
        (Tensor(1), int),
        ([1, 1], int),
        ([Tensor(1), Tensor(1)], int),
        ((1, 1), int),
        ((Tensor("1"), Tensor("2")), str),
        ({"a": 1}, int),
        ({"a": Tensor(1.0)}, float),
        (deque([1, 2]), int),
    ),
)
def test_get_arg_type(data, ref):
    assert _get_arg_type(data) == ref


@pytest.mark.parametrize(
    "data, ref",
    (
        (float, [float]),
        (float | str, [float, str]),
        (Union[float, str], [float, str]),  # noqa
        (list[float], [float]),
        (list[float | str], [float, str]),
        (dict[str, int], [int]),
        (dict[str, float | int], [float, int]),
    ),
)
def test_get_register_types(data, ref):
    assert _get_register_types(data) == ref
