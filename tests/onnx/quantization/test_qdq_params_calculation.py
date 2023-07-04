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
from typing import Dict

import numpy as np
import onnx
import pytest

from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from tests.onnx.conftest import ONNX_TEST_ROOT
from tests.onnx.models import LinearModel
from tests.onnx.quantization.common import min_max_quantize_model
from tests.shared.helpers import compare_stats
from tests.shared.helpers import load_json

REFERENCE_SCALES_DIR = ONNX_TEST_ROOT / "data" / "reference_scales"


def get_q_nodes_params(model: onnx.ModelProto) -> Dict[str, np.ndarray]:
    output = {}
    onnx_graph = ONNXGraph(model)
    for node in onnx_graph.get_all_nodes():
        if node.op_type == "QuantizeLinear":
            scale = onnx_graph.get_tensor_value(node.input[1])
            zero_point = onnx_graph.get_tensor_value(node.input[2])
            output[node.name] = {"scale": scale, "zero_point": zero_point}
    return output


@pytest.mark.parametrize(
    "overflow_fix",
    [OverflowFix.DISABLE, OverflowFix.ENABLE, OverflowFix.FIRST_LAYER],
    ids=[OverflowFix.DISABLE.value, OverflowFix.ENABLE.value, OverflowFix.FIRST_LAYER.value],
)
def test_overflow_fix_scales(overflow_fix):
    model = LinearModel()
    quantized_model = min_max_quantize_model(
        model.onnx_model,
        quantization_params={"advanced_parameters": AdvancedQuantizationParameters(overflow_fix=overflow_fix)},
    )
    q_nodes_params = get_q_nodes_params(quantized_model)

    ref_stats_name = model.path_ref_graph.split(".")[0] + f"_overflow_fix_{overflow_fix.value}.json"
    ref_stats_path = REFERENCE_SCALES_DIR / ref_stats_name

    # Unkomment lines below to generate reference for new models.
    # from tests.shared.helpers import dump_to_json
    # dump_to_json(ref_stats_path, q_nodes_params)

    ref_nodes_params = load_json(ref_stats_path)
    params = ["scale", "zero_point"]
    compare_stats(ref_nodes_params, q_nodes_params, params)
