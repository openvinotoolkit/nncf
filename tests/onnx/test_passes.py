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

from nncf.onnx.graph.passes import apply_preprocess_passes
from tests.onnx.models import build_matmul_model_with_nop_cast


def test_apply_preprocess_passes():
    model = build_matmul_model_with_nop_cast()
    before_nodes = [node.name for node in model.graph.node]
    preprocessed_model = apply_preprocess_passes(model)
    after_nodes = [node.name for node in preprocessed_model.graph.node]

    assert set(after_nodes) - set(before_nodes) == set()
    assert set(before_nodes) - set(after_nodes) == set(["cast"])
