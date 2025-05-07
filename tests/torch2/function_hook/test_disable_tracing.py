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

from nncf.torch import disable_tracing
from nncf.torch.function_hook.graph.build_graph_mode import build_graph
from nncf.torch.function_hook.graph.graph_visualization import to_pydot
from nncf.torch.function_hook.hook_executor_mode import FunctionHookMode
from nncf.torch.function_hook.hook_executor_mode import disable_function_hook_mode
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import wrap_model
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch2.utils import compare_with_reference_file

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "disable_tracing"


def test_disable_function_hook_mode():
    model = wrap_model(nn.Conv2d(1, 1, 1))
    with FunctionHookMode(model=model, hook_storage=get_hook_storage(model)) as ctx:
        assert ctx.enabled
        with disable_function_hook_mode():
            assert not ctx.enabled
        assert ctx.enabled


class ModelNoTrace(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x + 1
        x = self.foo(x)
        x = x - 1
        return x

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
