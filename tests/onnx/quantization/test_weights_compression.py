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
import numpy as np
import onnx
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper

from nncf.quantization import compress_weights


def create_model():
    # Define the model's input and output tensors.
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [100, 1280])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [100, 1280])

    nodes = []
    initializers = []

    # Create 10 MatMul nodes with ReLU activations.
    for i in range(10):
        w_size = [1280, 1280]
        weight_data = np.random.rand(*w_size).astype(np.float32)
        weight_initializer = numpy_helper.from_array(weight_data, name=f"weight_{i}")
        initializers.append(weight_initializer)

        input_name = "input" if i == 0 else f"relu_{i - 1}"
        matmul_output_name = f"matmul_{i}"
        relu_output_name = "output" if i == 9 else f"relu_{i}"

        matmul_node = helper.make_node("MatMul", inputs=[input_name, f"weight_{i}"], outputs=[matmul_output_name])
        nodes.append(matmul_node)

        relu_node = helper.make_node("Relu", inputs=[matmul_output_name], outputs=[relu_output_name])
        nodes.append(relu_node)

    # Build the graph.
    graph_def = helper.make_graph(
        nodes=nodes,
        name="StackedMatMulGraphWithReLU",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers,
    )

    # Create the model and set the opset version to 21.
    model_def = helper.make_model(graph_def, producer_name="synthetic-onnx-model")
    model_def.opset_import[0].version = 21

    return model_def


def test_wc():
    model = create_model()
    model = compress_weights(model)
    onnx.save_model(model, "compressed_model.onnx")
