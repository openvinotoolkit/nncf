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
from dataclasses import dataclass

import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import TensorProto
from onnx import helper
from onnx import numpy_helper
from onnxruntime import InferenceSession
from packaging import version

from nncf import CompressWeightsMode
from nncf.onnx.graph.onnx_helper import get_edge_shape
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.quantization import compress_weights


def create_model(opset_version=21):
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
    model_def.opset_import[0].version = opset_version

    return model_def


@dataclass
class WeightTypeCounter:
    int4: int = 0
    uint4: int = 0
    int8: int = 0
    uint8: int = 0


def calculate_numbers_of_quantized_weights(model: onnx.ModelProto) -> WeightTypeCounter:
    counter = WeightTypeCounter()
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            x = get_tensor(model, node.input[0])
            if x.data_type == TensorProto.INT8:
                counter.int8 += 1
            elif x.data_type == TensorProto.UINT8:
                counter.uint8 += 1
            elif x.data_type == TensorProto.INT4:
                counter.int4 += 1
            elif x.data_type == TensorProto.UINT4:
                counter.uint4 += 1
            else:
                msg = f"Unexpected data type: {x.data_type}"
                raise RuntimeError(msg)
    return counter


@pytest.mark.parametrize(
    "mode, reference_counter",
    [
        [CompressWeightsMode.INT8_ASYM, WeightTypeCounter(int4=0, uint4=0, int8=0, uint8=10)],
        [CompressWeightsMode.INT8_SYM, WeightTypeCounter(int4=0, uint4=0, int8=10, uint8=0)],
        [CompressWeightsMode.INT4_ASYM, WeightTypeCounter(int4=0, uint4=9, int8=0, uint8=1)],
        [CompressWeightsMode.INT4_SYM, WeightTypeCounter(int4=9, uint4=0, int8=0, uint8=1)],
    ],
)
def test_numbers_of_quantized_weights(mode, reference_counter):
    model = create_model()
    model = compress_weights(model, mode)
    counter = calculate_numbers_of_quantized_weights(model)
    assert counter == reference_counter


@pytest.mark.parametrize(
    "mode_weight_type",
    [(CompressWeightsMode.INT8_SYM, TensorProto.INT8)],
)
def test_correct_dequantizelinear_int8(mode_weight_type):
    mode, expected_weight_type = mode_weight_type
    model = create_model()
    model = compress_weights(model, mode)

    dq_cnt = 0
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            assert node.input[0] == f"weight_{dq_cnt}_quantized"
            for attr in node.attribute:
                if attr.name == "axis":
                    assert attr.i == -1
                if attr.name == "block_size":
                    assert attr.i == 0

            weight_tensor = get_tensor(model, node.input[0])
            assert weight_tensor.data_type == expected_weight_type
            assert get_edge_shape(weight_tensor) == [1280, 1280]

            weight_tensor = get_tensor(model, node.input[1])
            assert weight_tensor.data_type == TensorProto.FLOAT
            assert get_edge_shape(weight_tensor) == [1280]

            assert len(node.input) == 2
            dq_cnt += 1


@pytest.mark.parametrize(
    "mode_weight_type",
    [(CompressWeightsMode.INT8_ASYM, TensorProto.UINT8)],
)
def test_correct_dequantizelinear_uint8(mode_weight_type):
    mode, expected_weight_type = mode_weight_type
    model = create_model()
    model = compress_weights(model, mode)

    dq_cnt = 0
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            assert node.input[0] == f"weight_{dq_cnt}_quantized"
            for attr in node.attribute:
                if attr.name == "axis":
                    assert attr.i == -1
                if attr.name == "block_size":
                    assert attr.i == 0

            weight_tensor = get_tensor(model, node.input[0])
            assert weight_tensor.data_type == expected_weight_type
            assert get_edge_shape(weight_tensor) == [1280, 1280]

            weight_tensor = get_tensor(model, node.input[1])
            assert weight_tensor.data_type == TensorProto.FLOAT
            assert get_edge_shape(weight_tensor) == [1280]

            zero_point_tensor = get_tensor(model, node.input[2])
            assert zero_point_tensor.data_type == expected_weight_type
            assert get_edge_shape(zero_point_tensor) == [1280]

            dq_cnt += 1


@pytest.mark.parametrize(
    "mode_weight_type",
    [
        (CompressWeightsMode.INT4_SYM, TensorProto.INT4),
    ],
)
@pytest.mark.parametrize(
    "group_size",
    [1, 4, 8, 128, 1280],
)
def test_correct_dequantizelinear_int4(mode_weight_type, group_size):
    mode, expected_weight_type = mode_weight_type
    model = create_model()
    model = compress_weights(model, mode, group_size=group_size, all_layers=True)

    dq_cnt = 0
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            assert node.input[0] == f"weight_{dq_cnt}_quantized"
            for attr in node.attribute:
                if attr.name == "axis":
                    assert attr.i == 0
                if attr.name == "block_size":
                    assert attr.i == group_size

            weight_tensor = get_tensor(model, node.input[0])
            assert weight_tensor.data_type == expected_weight_type
            assert get_edge_shape(weight_tensor) == [1280, 1280]

            weight_tensor = get_tensor(model, node.input[1])
            assert weight_tensor.data_type == TensorProto.FLOAT
            assert get_edge_shape(weight_tensor) == [1280 // group_size, 1280]
            assert len(node.input) == 2
            dq_cnt += 1


@pytest.mark.parametrize(
    "mode_weight_type",
    [
        (CompressWeightsMode.INT4_ASYM, TensorProto.UINT4),
    ],
)
@pytest.mark.parametrize(
    "group_size",
    [1, 4, 8, 128, 1280],
)
def test_correct_dequantizelinear_uint4(mode_weight_type, group_size):
    mode, expected_weight_type = mode_weight_type
    model = create_model()
    model = compress_weights(model, mode, group_size=group_size, all_layers=True)

    dq_cnt = 0
    for node in model.graph.node:
        if node.op_type == "DequantizeLinear":
            assert node.input[0] == f"weight_{dq_cnt}_quantized"
            for attr in node.attribute:
                if attr.name == "axis":
                    assert attr.i == 0
                if attr.name == "block_size":
                    assert attr.i == group_size

            weight_tensor = get_tensor(model, node.input[0])
            assert weight_tensor.data_type == expected_weight_type
            assert get_edge_shape(weight_tensor) == [1280, 1280]

            scale_tensor = get_tensor(model, node.input[1])
            assert scale_tensor.data_type == TensorProto.FLOAT
            assert get_edge_shape(scale_tensor) == [1280 // group_size, 1280]

            zero_point_tensor = get_tensor(model, node.input[2])
            assert zero_point_tensor.data_type == expected_weight_type
            assert get_edge_shape(zero_point_tensor) == [1280 // group_size, 1280]
            dq_cnt += 1


@pytest.mark.xfail(
    version.parse(onnx.__version__) >= version.parse("1.18.0"),
    reason="onnxruntime not support default IR for onnx==1.18.0",
)
@pytest.mark.parametrize(
    "mode",
    [
        CompressWeightsMode.INT8_ASYM,
        CompressWeightsMode.INT8_SYM,
        CompressWeightsMode.INT4_ASYM,
        CompressWeightsMode.INT4_SYM,
    ],
)
def test_compression_with_inference(mode):
    model = create_model()
    model = compress_weights(model, mode)
    onnx.checker.check_model(model)
    input_data = np.random.rand(100, 1280).astype(np.float32)
    session = InferenceSession(model.SerializeToString())
    session.run(None, {"input": input_data})


def test_matmulnbits():
    rtol = 1e-5
    if version.parse(onnxruntime.__version__) < version.parse("1.21.1"):
        rtol = 1e-3

    np.random.seed(42)
    model_opset21 = create_model()

    np.random.seed(42)
    model_opset19 = create_model(opset_version=19)

    compressed_model_opset21 = compress_weights(model_opset21, mode=CompressWeightsMode.INT4_SYM, group_size=16)
    compressed_model_opset19 = compress_weights(model_opset19, mode=CompressWeightsMode.INT4_SYM, group_size=16)

    dummy_input = np.random.rand(100, 1280).astype(np.float32)

    sess21 = InferenceSession(compressed_model_opset21.SerializeToString())
    sess19 = InferenceSession(compressed_model_opset19.SerializeToString())

    output21 = sess21.run(None, {"input": dummy_input})[0]
    output19 = sess19.run(None, {"input": dummy_input})[0]

    assert np.allclose(output21, output19, rtol=rtol, atol=1e-6)
