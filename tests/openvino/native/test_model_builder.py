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

from pathlib import Path

import openvino.runtime as ov
import pytest

from nncf.openvino.graph.model_builder import OVModelBuilder
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.native.common import get_actual_reference_for_current_openvino
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import DynamicModel
from tests.openvino.native.models import FPModel
from tests.openvino.native.models import LinearModel

REFERENCE_GRAPHS_DIR = Path("reference_graphs") / "original_nncf_graph"

MODEL_BUILDER = OVModelBuilder()

FAST_BC_CASES = [
    {
        "model": ConvModel(),
        "input_ids": [("Conv", 0)],
        "output_ids": [("Conv", 0)],
    },
    {
        "model": FPModel(const_dtype=ov.Type.bf16),
        "input_ids": [("MatMul", 0)],
        "output_ids": [("MatMul", 0)],
    },
    {
        "model": LinearModel(),
        "input_ids": [("MatMul", 0)],
        "output_ids": [("MatMul", 0)],
    },
    {
        "model": DynamicModel(),
        "input_ids": [("Conv", 0)],
        "output_ids": [("Conv", 0)],
    },
]

TESTING_MODELS_DATA = FAST_BC_CASES


@pytest.mark.parametrize("model_data", TESTING_MODELS_DATA)
def test_model_building(model_data):
    model_to_test = model_data["model"]
    model = model_to_test.ov_model

    node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
    built_model = MODEL_BUILDER.build(
        input_ids=model_data["input_ids"], output_ids=model_data["output_ids"], node_mapping=node_mapping
    )

    path_to_dot = get_actual_reference_for_current_openvino(
        REFERENCE_GRAPHS_DIR / f"built_{model_to_test.ref_graph_name}"
    )
    compare_nncf_graphs(built_model, path_to_dot)
