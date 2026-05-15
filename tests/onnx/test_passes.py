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
import numpy as np
import onnx

import nncf
from nncf.onnx.graph.passes import apply_preprocess_passes
from nncf.onnx.graph.passes import compress_quantize_weights_transformation
from nncf.onnx.quantization.backend_parameters import BackendParameters
from tests.onnx.common import ModelBuilder
from tests.onnx.models import build_matmul_model_with_nop_cast


def test_apply_preprocess_passes():
    model = build_matmul_model_with_nop_cast()
    before_nodes = [node.name for node in model.graph.node]
    preprocessed_model = apply_preprocess_passes(model)
    after_nodes = [node.name for node in preprocessed_model.graph.node]

    assert set(after_nodes) - set(before_nodes) == set()
    assert set(before_nodes) - set(after_nodes) == set(["cast"])


def _build_model():
    w = np.array([[0.1, 0.3, 0.2, -0.1], [-0.9, 0.1, 0.5, -0.3], [0.0, -0.1, -0.4, -0.9]], dtype=np.float32)

    b = np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32)

    mb = ModelBuilder()
    x = mb.add_input("X", (2, 3))
    x = mb.add_gemm(x, w.shape, weight_data=w, bias_data=b)
    mb.add_output(x, (2, 4))
    return mb.build(opset_version=19, ir_version=9)


def check_operation_count(model: onnx.ModelProto, op_type_to_count: dict[str, int]):
    count = {}
    for node in model.graph.node:
        if node.op_type in op_type_to_count:
            count[node.op_type] = count.get(node.op_type, 0) + 1
    assert count == op_type_to_count


def test_compress_quantize_weights_transformation():
    model = _build_model()

    x = np.array([[0.2, -0.1, 0.9], [-0.1, -0.9, 0.5]], dtype=np.float32)

    # Prepare quantized model without weight compression
    calibration_dataset = nncf.Dataset([{"X": x}])
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        advanced_parameters=nncf.AdvancedQuantizationParameters(
            backend_params={BackendParameters.COMPRESS_WEIGHTS: False}
        ),
    )

    check_operation_count(quantized_model, {"QuantizeLinear": 2, "DequantizeLinear": 2})
    compress_quantize_weights_transformation(quantized_model)
    check_operation_count(quantized_model, {"QuantizeLinear": 1, "DequantizeLinear": 2})
