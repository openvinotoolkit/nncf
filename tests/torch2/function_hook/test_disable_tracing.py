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

import torch
from torch import nn
from torch.overrides import _get_current_function_mode_stack

from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph
from nncf.experimental.torch2.function_hook.graph.graph_visualization import to_pydot
from nncf.experimental.torch2.function_hook.hook_executor_mode import FunctionHookMode
from nncf.experimental.torch2.function_hook.wrapper import wrap_model
from nncf.torch import disable_tracing
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch2.utils import compare_with_reference_file

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "disable_tracing"


class ModelNoTrace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x + 1
        return self.foo(x)

    def foo(self, x):
        mode = _get_current_function_mode_stack()
        assert len(mode) == 1
        assert isinstance(mode[0], FunctionHookMode)
        assert not mode[0].enabled
        return x - 1


disable_tracing(ModelNoTrace.foo)


def test_build_graph_with_disable_tracing(regen_ref_data):
    model = wrap_model(ModelNoTrace())
    graph = build_graph(model, torch.randn(1, 1))
    dot_graph = to_pydot(graph)
    compare_with_reference_file(str(dot_graph), REF_DIR / "graph.dot", regen_ref_data)
