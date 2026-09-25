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
import pytest
import torch

import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.magnitude.pattern import SparsityPattern
from nncf.torch.function_hook.pruning.magnitude.pattern import get_sparsity_pattern
from nncf.torch.function_hook.pruning.magnitude.pattern import get_structured_pattern_groups
from nncf.torch.graph.operator_metatypes import PTLinearMetatype


def test_sparsity_pattern_str():
    assert str(SparsityPattern(m=2, n=4)) == "2:4"


@pytest.mark.parametrize("m, n", ((0, 4), (-1, 4), (2, 0), (2, -4), (4, 4), (5, 4)))
def test_sparsity_pattern_invalid(m: int, n: int):
    with pytest.raises(nncf.ValidationError, match="m should be less than n|should be positive"):
        SparsityPattern(m=m, n=n)


def test_get_sparsity_pattern_for_structured_mode():
    pattern = get_sparsity_pattern(PruneMode.STRUCTURED_MAGNITUDE_2_4)
    assert pattern.m == 2
    assert pattern.n == 4


def test_get_sparsity_pattern_for_unsupported_mode():
    with pytest.raises(nncf.InternalError, match="does not define a structured sparsity pattern"):
        get_sparsity_pattern(PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL)


def test_get_structured_pattern_groups_linear():
    # Linear weight of shape (out_channels, in_features) with in_features = 8
    data = torch.arange(96, dtype=torch.float32).reshape(12, 8)
    pattern = SparsityPattern(m=2, n=4)
    groups = get_structured_pattern_groups(data, PTLinearMetatype, 1, pattern)
    assert groups.shape == (12, 2, 4)
    # Groups of consecutive weights within one output channel
    assert torch.equal(groups[0, 0], data[0, :4])
    assert torch.equal(groups[0, 1], data[0, 4:])
    assert torch.equal(groups[7, 1], data[7, 4:])


def test_get_structured_pattern_groups_not_divisible():
    data = torch.ones(4, 7)
    pattern = SparsityPattern(m=2, n=4)
    with pytest.raises(nncf.ValidationError, match="not divisible"):
        get_structured_pattern_groups(data, PTLinearMetatype, 1, pattern)
