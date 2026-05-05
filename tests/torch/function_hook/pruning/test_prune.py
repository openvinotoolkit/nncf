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

from dataclasses import dataclass

import pytest

import nncf
from nncf.scopes import IgnoredScope
from nncf.torch.function_hook.pruning.prune_model import get_prunable_parameters
from nncf.torch.graph.graph import PTNNCFGraph
from tests.torch.function_hook.pruning.helpers import TwoConvModel


@pytest.fixture(scope="session")
def two_conv_graph() -> PTNNCFGraph:
    model = TwoConvModel()
    return nncf.build_graph(model, example_input=model.get_example_inputs())


@dataclass
class IgnoredScopeParam:
    test_name: str
    ignored_scope: IgnoredScope
    expected_params: set[str]

    def __str__(self):
        return self.test_name


@pytest.mark.parametrize(
    "param",
    (
        IgnoredScopeParam(
            test_name="none",
            ignored_scope=None,
            expected_params={"conv1.weight", "conv2.weight"},
        ),
        IgnoredScopeParam(
            test_name="weight_name_1",
            ignored_scope=IgnoredScope(names=["conv1.weight"]),
            expected_params={"conv2.weight"},
        ),
        IgnoredScopeParam(
            test_name="weight_name_2",
            ignored_scope=IgnoredScope(names=["conv2.weight"]),
            expected_params={"conv1.weight"},
        ),
        IgnoredScopeParam(
            test_name="op_name",
            ignored_scope=IgnoredScope(names=["conv1/conv2d/0"]),
            expected_params={"conv2.weight"},
        ),
        IgnoredScopeParam(
            test_name="pattern_conv",
            ignored_scope=IgnoredScope(patterns=["^conv1/conv2d/.*"]),
            expected_params={"conv2.weight"},
        ),
        IgnoredScopeParam(
            test_name="pattern_const",
            ignored_scope=IgnoredScope(patterns=[".*2.weight"]),
            expected_params={"conv1.weight"},
        ),
    ),
    ids=str,
)
def test_ignore_scope_for_prunable_parameters(param: IgnoredScopeParam, two_conv_graph: PTNNCFGraph):
    prunable_parameters = get_prunable_parameters(two_conv_graph, param.ignored_scope)
    assert prunable_parameters == param.expected_params
