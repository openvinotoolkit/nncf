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


import pytest

from nncf.experimental.torch2.function_hook.graph.build_graph_mode import build_graph
from nncf.experimental.torch2.function_hook.graph.graph_visualization import PydotStyleTemplate
from nncf.experimental.torch2.function_hook.graph.graph_visualization import to_pydot
from tests.cross_fw.shared.paths import TEST_ROOT
from tests.torch2.function_hook import helpers
from tests.torch2.utils import compare_with_reference_file

REF_DIR = TEST_ROOT / "torch2" / "data" / "function_hook" / "graph_visualization"


@pytest.mark.parametrize("style", PydotStyleTemplate)
def test_to_pydot(style, regen_ref_data: bool):
    model = helpers.get_wrapped_simple_model_with_hook()
    graph = build_graph(model, model.get_example_inputs())

    dot_graph = to_pydot(graph, style_template=style)
    ref_file = REF_DIR / f"to_pydot_style_{style}.dot"

    compare_with_reference_file(str(dot_graph), ref_file, regen_ref_data)
