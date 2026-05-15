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
class ParamIgnoredScope:
    name: str
    ignored_scope: IgnoredScope
    expected_params: set[str]

    def __str__(self) -> str:
        return self.name


@pytest.mark.parametrize(
    "param",
    (
        ParamIgnoredScope("empty", IgnoredScope(), {"conv1.weight", "conv2.weight"}),
        ParamIgnoredScope("weight_name_1", IgnoredScope(names=["conv1.weight"]), {"conv2.weight"}),
        ParamIgnoredScope("weight_name_2", IgnoredScope(names=["conv2.weight"]), {"conv1.weight"}),
        ParamIgnoredScope("op_name", IgnoredScope(names=["conv1/conv2d/0"]), {"conv2.weight"}),
        ParamIgnoredScope("pattern_conv", IgnoredScope(patterns=["^conv1/conv2d/.*"]), {"conv2.weight"}),
        ParamIgnoredScope("pattern_const", IgnoredScope(patterns=[".*2.weight"]), {"conv1.weight"}),
    ),
    ids=str,
)
def test_ignore_scope_for_prunable_parameters(param: ParamIgnoredScope, two_conv_graph: PTNNCFGraph):
    prunable_parameters = get_prunable_parameters(two_conv_graph, param.ignored_scope)
    assert prunable_parameters == param.expected_params
