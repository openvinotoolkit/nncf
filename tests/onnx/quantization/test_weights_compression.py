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

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Callable, Optional

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
from nncf.common.factory import EngineFactory
from nncf.common.factory import build_graph
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.node_utils import get_input_edges_mapping
from nncf.onnx.graph.onnx_helper import get_edge_shape
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.quantization import compress_weights
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from tests.cross_fw.test_templates.template_test_weights_compression import TemplateWeightCompression
from tests.onnx.common import ModelBuilder


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
    model = compress_weights(model, mode=mode)
    counter = calculate_numbers_of_quantized_weights(model)
    assert counter == reference_counter


@pytest.mark.parametrize(
    "mode_weight_type",
    [(CompressWeightsMode.INT8_SYM, TensorProto.INT8)],
)
def test_correct_dequantizelinear_int8(mode_weight_type):
    mode, expected_weight_type = mode_weight_type
    model = create_model()
    model = compress_weights(model, mode=mode)

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
    model = compress_weights(model, mode=mode)

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
    model = compress_weights(model, mode=mode, group_size=group_size, all_layers=True)

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
    model = compress_weights(model, mode=mode, group_size=group_size, all_layers=True)

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
    model = compress_weights(model, mode=mode)
    onnx.checker.check_model(model)
    input_data = np.random.rand(100, 1280).astype(np.float32)
    session = InferenceSession(model.SerializeToString())
    session.run(None, {"input": input_data})


@pytest.mark.xfail(
    version.parse(onnx.__version__) >= version.parse("1.18.0"),
    reason="onnxruntime not support default IR for onnx==1.18.0",
)
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


@pytest.mark.parametrize("trans_b", [0, 1])
def test_matmulnbits_gemm(trans_b: int):
    # Build the model with a single Gemm operation
    np.random.seed(42)

    w = np.random.rand(1280, 10).astype(np.float32)
    if trans_b:
        w = w.T
    b = np.random.rand(10).astype(np.float32)

    mb = ModelBuilder()
    x = mb.add_input("input", (1, 1280))
    x = mb.add_gemm(x, shape=w.shape, weight_data=w, bias_data=b, trans_b=trans_b)

    mb.add_output(x, (1, 10))

    model_opset19 = mb.build(opset_version=19)
    model_opset21 = mb.build(opset_version=21)

    rtol = 1e-5
    if version.parse(onnxruntime.__version__) < version.parse("1.21.1"):
        rtol = 1e-3

    compressed_model_opset21 = compress_weights(model_opset21, mode=CompressWeightsMode.INT4_SYM, group_size=64)
    compressed_model_opset19 = compress_weights(model_opset19, mode=CompressWeightsMode.INT4_SYM, group_size=64)

    dummy_input = np.random.rand(1, 1280).astype(np.float32)

    sess21 = InferenceSession(compressed_model_opset21.SerializeToString())
    sess19 = InferenceSession(compressed_model_opset19.SerializeToString())

    output21 = sess21.run(None, {"input": dummy_input})[0]
    output19 = sess19.run(None, {"input": dummy_input})[0]

    assert np.allclose(output21, output19, rtol=rtol, atol=1e-6)


class TestONNXTemplateWeightCompression(TemplateWeightCompression):
    @staticmethod
    def cast_to(x: np.ndarray, dtype: TensorDataType) -> np.ndarray:
        if dtype is TensorDataType.float32:
            return x.astype(np.float32)
        if dtype is TensorDataType.float16:
            return x.astype(np.float16)
        raise NotImplementedError

    @staticmethod
    def get_matmul_model() -> onnx.ModelProto:
        """
        Builds a model to be used in the TemplateWeightCompression.test_data_based_criterion() test.
        """
        mb = ModelBuilder()
        x = mb.add_input("input", (1, 3, 3))
        output = mb.add_output("output", (1, 3, 3))
        weights_data = np.eye(3, dtype=np.float32) * 255
        mb.add_matmul(x, shape=weights_data.shape, output=output, data=weights_data)
        return mb.build()

    @staticmethod
    def get_RoPE_model() -> onnx.ModelProto:
        """
        Builds a model to be used in the TemplateWeightCompression.test_rope_weight_compression() test.
        """
        mb = ModelBuilder()

        x = mb.add_input("input", (1, 10))
        x = mb.add_unsqueeze(x, axes=(2,))
        x = mb.add_matmul(x, shape=(1, 5))
        x = mb.add_transpose(x, perm=[0, 2, 1])
        x = mb.add_concat([x], axis=-1)
        x1 = mb.add_sin(x)
        x2 = mb.add_cos(x)

        mb.add_output(x1, (1, 5, 10))
        mb.add_output(x2, (1, 5, 10))

        return mb.build()

    @staticmethod
    def get_SAM_PE_model() -> onnx.ModelProto:
        """
        Builds a model to be used in the TemplateWeightCompression.test_sam_pe_weight_compression() test.
        """
        mb = ModelBuilder()

        x = mb.add_input("input", (-1, -1, -1, 2))
        x = mb.add_matmul(x, shape=(2, 128))
        x = mb.add_mul_const(x, shape=(1,), data=np.array([2 * np.pi], np.float32))
        x1 = mb.add_sin(x)
        x2 = mb.add_cos(x)
        x = mb.add_concat([x1, x2], axis=-1)

        mb.add_output(x, (-1, -1, -1, 256))

        return mb.build()

    @staticmethod
    def get_sequential_matmul_model() -> onnx.ModelProto:
        """
        Builds a model to be used in the TemplateWeightCompression.test_mixed_precision() test.
        """
        mb = ModelBuilder()
        x = mb.add_input("input", (1, 4, 4))
        output = mb.add_output("output", (1, 4, 4))

        main_values = [10000, 1000, 1, 10, 10000]
        for i, main_value in enumerate(main_values):
            weights_data = np.arange(0, 16).reshape(4, 4).astype(np.float32)
            weights_data[-1, -1] = main_value
            weights_data = weights_data.T
            x = mb.add_matmul(x, shape=weights_data.shape, output=output if i == 4 else None, data=weights_data)

        return mb.build(opset_version=21)

    @staticmethod
    def to_tensor(x: np.ndarray) -> np.ndarray:
        return np.array(x)

    @staticmethod
    def check_weights(model: onnx.ModelProto, ref_ids: list[int]) -> None:
        names = {i.name for i in model.graph.initializer if i.data_type == onnx.TensorProto.INT4}
        low_precision_nodes = {f"W_{i}_quantized" for i in ref_ids}
        assert low_precision_nodes == names

    @staticmethod
    def get_not_supported_algorithms() -> list[str]:
        return ["gptq", "lora_correction"]

    @staticmethod
    def wrap_model(model: onnx.ModelProto, data: Any) -> onnx.ModelProto:
        return model

    @staticmethod
    def get_model_for_test_scale_estimation() -> onnx.ModelProto:
        """
        Builds a model to be used in the following tests:
            - TemplateWeightCompression.test_scale_estimation()
            - TemplateWeightCompression.test_scale_estimation_outlier_channel_has_lowest_error()
        tests.
        """
        mb = ModelBuilder()
        x = mb.add_input("input", (1, 4, 8))
        output = mb.add_output("output", (1, 4, 16))
        weights = np.arange(0, 16 * 8, dtype=np.float32).reshape(16, 8).T
        mb.add_matmul(x, shape=(8, 16), output=output, data=weights)

        return mb.build(opset_version=21)

    @staticmethod
    def get_moe_model_for_test_scale_estimation() -> onnx.ModelProto:
        num_experts = 2
        hidden_dim = 8
        out_dim = 16
        seq_len = 4

        mb = ModelBuilder()
        x = mb.add_input("input", (num_experts, seq_len, hidden_dim))
        output = mb.add_output("output", (num_experts, seq_len, out_dim))

        weights = np.arange(0, num_experts * hidden_dim * out_dim, dtype=np.float32)
        weights = weights.reshape(num_experts, hidden_dim, out_dim)

        mb.add_matmul(x, shape=(num_experts, hidden_dim, out_dim), output=output, data=weights)

        return mb.build(opset_version=21)

    @staticmethod
    def get_scale_estimation_ref(check_sampling_activation_stats_flow):
        return (
            np.array(
                [
                    [[0.473328]],
                    [[0.929023]],
                    [[1.446527]],
                    [[1.920595]],
                    [[2.517054]],
                    [[3.030102]],
                    [[3.584279]],
                    [[4.043509]],
                    [[4.620008]],
                    [[5.165322]],
                    [[5.710637]],
                    [[6.122581]],
                    [[6.655914]],
                    [[7.237174]],
                    [[7.722580]],
                    [[8.255914]],
                ]
            ).T,
            np.array(
                [
                    [[0.47344488]],
                    [[0.9287766]],
                    [[1.4463282]],
                    [[1.920052]],
                    [[2.5167778]],
                    [[3.02987]],
                    [[3.5842714]],
                    [[4.0429296]],
                    [[4.619769]],
                    [[5.165224]],
                    [[5.7106786]],
                    [[6.121212]],
                    [[6.654546]],
                    [[7.2366524]],
                    [[7.7212124]],
                    [[8.254545]],
                ]
            ).T,
        )[check_sampling_activation_stats_flow]

    @staticmethod
    def get_moe_scale_estimation_ref(check_sampling_activation_stats_flow):
        return (
            np.array(
                [
                    [
                        [
                            [
                                7.5732,
                                7.4667,
                                7.4667,
                                7.4667,
                                7.4667,
                                7.2602,
                                7.4667,
                                7.4667,
                                7.4667,
                                7.4667,
                                7.3083,
                                7.8467,
                                7.2233,
                                7.2715,
                                7.4205,
                                7.4667,
                            ]
                        ]
                    ],
                    [
                        [
                            [
                                14.8205,
                                14.9032,
                                14.9858,
                                15.0685,
                                15.1512,
                                14.3400,
                                14.4173,
                                14.4945,
                                14.5718,
                                14.6491,
                                14.7264,
                                14.8037,
                                14.8810,
                                14.9583,
                                15.0355,
                                15.1128,
                            ]
                        ]
                    ],
                ]
            ),
            np.array(
                [
                    [
                        [
                            [
                                7.575118,
                                7.4666667,
                                7.4666667,
                                7.4666667,
                                7.4666667,
                                7.254837,
                                7.4666667,
                                7.4666667,
                                7.4666667,
                                7.4666667,
                                7.495066,
                                7.850108,
                                7.219489,
                                7.2685375,
                                7.418597,
                                7.4666667,
                            ]
                        ]
                    ],
                    [
                        [
                            [
                                14.820066,
                                14.902746,
                                14.985427,
                                15.068108,
                                15.150787,
                                14.3391285,
                                14.416424,
                                14.493721,
                                14.571016,
                                14.648311,
                                14.725608,
                                14.802904,
                                14.8801985,
                                14.957496,
                                15.034791,
                                15.112087,
                            ]
                        ]
                    ],
                ]
            ),
        )[check_sampling_activation_stats_flow]

    @staticmethod
    def get_orig_weight(model: onnx.ModelProto) -> Tensor:
        return Tensor(get_tensor_value(model, "W_0"))

    @pytest.fixture(params=(CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM))
    def int4_mode(self, request):
        return request.param

    @staticmethod
    def get_decompressed_weight(compressed_model: onnx.ModelProto, input: np.ndarray):
        graph = build_graph(compressed_model)
        mapping = get_input_edges_mapping(graph)
        transformation_layout = TransformationLayout()

        transformation_layout.register(
            ONNXOutputInsertionCommand(
                ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, "W_0_DequantizeLinear", 0), mapping
            )
        )
        mt = ONNXModelTransformer(compressed_model)
        transformed_model = mt.transform(transformation_layout)

        onnx.save(transformed_model, "transformed_model.onnx")

        engine = EngineFactory.create(transformed_model)
        outputs = engine.infer({"input": input})
        return Tensor(outputs["W_0_dequantized"])

    @staticmethod
    def get_awq_act_model(is_3d_weights: bool, with_multiply: bool, n_layers: int) -> onnx.ModelProto:
        """
        Builds a model to be used in the following tests:
            - TemplateWeightCompression.test_call_max_var_criterion_with_dataset_by_default_awq_act_matmul()
            - TemplateWeightCompression.test_data_free_awq()
        tests.
        """
        mb = ModelBuilder()

        weight_shape = (8, 8)
        if is_3d_weights:
            # The first and last dimension are later transposed
            weight_shape = (8, 8, 2)

        data = 0.01 * np.arange(0, reduce(mul, weight_shape, 1)).reshape(weight_shape) + 0.05
        data = data.astype(np.float32).T

        x = mb.add_input("input", (2, 8, 8))
        output = mb.add_output("output", (2, 8, 8))

        x = mb.add_matmul(x, shape=data.shape, data=data)
        for _ in range(n_layers):
            a = mb.add_matmul(x, shape=data.shape, data=data)
            a = mb.add_relu(a)
            if with_multiply:
                b = mb.add_matmul(x, shape=data.shape, data=data)
                b = mb.add_selu(b)
                x = mb.add_mul(a, b)
            else:
                x = a
        mb.add_matmul(x, shape=data.shape, output=output, data=data)

        return mb.build(opset_version=21)

    @staticmethod
    def get_num_multiply_from_awq(model: onnx.ModelProto) -> int:
        awq_num = 0
        for node in model.graph.node:
            if node.op_type == "Mul" and "awq_mul" in node.name:
                awq_num += 1
        return awq_num

    @staticmethod
    def get_awq_model(is_3d_weights) -> onnx.ModelProto:
        """
        Builds a model to be used in the following tests:
            - TemplateWeightCompression.test_awq_with_ignored_scope()
            - TemplateWeightCompression.test_awq_scale_reference()
            - TemplateWeightCompression.test_error_message_for_invalid_group_size()
        tests.
        """
        mb = ModelBuilder()

        weight_shape = (8, 8)
        opset_version = 13
        if is_3d_weights:
            # The first and last dimension are later transposed
            weight_shape = (8, 8, 2)
            # 3D weights does not work due to no support in MatMulNBits which is used in opset_version < 21
            opset_version = 21

        x = mb.add_input("input", (2, None, 8))
        output = mb.add_output("output", (2, None, 8))

        w_data = 0.01 * np.arange(0, reduce(mul, weight_shape, 1), dtype=np.float32).reshape(weight_shape) + 0.05
        w_data = w_data.T

        num_blocks = 2
        for i in range(num_blocks):
            a = mb.add_matmul(x, shape=w_data.shape, data=w_data)
            b = mb.add_matmul(x, shape=w_data.shape, data=w_data)
            x = mb.add_mul(a, b)
            x = mb.add_matmul(x, shape=w_data.shape, output=output if i == num_blocks - 1 else None, data=w_data)

        return mb.build(opset_version=opset_version)

    @staticmethod
    def get_different_channel_size_model(channel_sizes: list[int]) -> onnx.ModelProto:
        """
        Builds a model to be used in the TemplateWeightCompression.test_group_size_fallback_modes() test.
        """
        mb = ModelBuilder()

        x = mb.add_input("input", (1, channel_sizes[0], channel_sizes[0]))
        output = mb.add_output("output", None)
        for i in range(1, len(channel_sizes) + 1):
            prev_channel_size = channel_sizes[i - 1]
            channel_size = channel_sizes[min(i, len(channel_sizes) - 1)]
            w_data = (
                np.arange(0, channel_size * prev_channel_size, dtype=np.float32)
                .reshape(channel_size, prev_channel_size)
                .T
            )
            x = mb.add_matmul(x, shape=w_data.shape, output=output if i == len(channel_sizes) else None, data=w_data)

        return mb.build(opset_version=21)

    @staticmethod
    def get_num_int4_nodes(model: onnx.ModelProto) -> int:
        num = 0
        for i in model.graph.initializer:
            if i.data_type in [onnx.TensorProto.UINT4, onnx.TensorProto.INT4]:
                num += 1
        return num

    @staticmethod
    def get_num_int4_group_sizes(model: onnx.ModelProto) -> dict[int, int]:
        num = defaultdict(int)
        for i in model.graph.initializer:
            if i.data_type in [onnx.TensorProto.UINT4, onnx.TensorProto.INT4]:
                shape = list(reversed(i.dims))
                num[shape[-1]] += 1
        return num

    @staticmethod
    def get_ignored_scope_name(is_3d_weights) -> str:
        return "MatMul_4"  # Zero-based indices (e.g., MatMul_0, MatMul_1, ...)

    @staticmethod
    def get_reference_for_test_awq_scale_reference(is_3d_weights) -> dict[str, Tensor]:
        return [
            {
                "MatMul_3": Tensor(
                    np.array(
                        [[1.4228648, 1.3474456, 1.1335096, 1.001522, 0.90938693, 0.84022623, 0.78575736, 0.7413683]],
                        dtype=np.float32,
                    ).T
                )
            },
            {
                "MatMul_3": Tensor(
                    np.array(
                        [
                            [
                                [1.119726],
                                [1.1012304],
                                [1.0438583],
                                [1.006067],
                                [0.97812414],
                                [0.95607865],
                                [0.9379444],
                                [0.922586],
                            ],
                            [
                                [0.99698645],
                                [0.9808075],
                                [0.9307146],
                                [0.8974796],
                                [0.87281394],
                                [0.8533093],
                                [0.8372402],
                                [0.82361573],
                            ],
                        ],
                        dtype=np.float32,
                    )
                )
            },
        ][is_3d_weights]

    @staticmethod
    def get_transform_func() -> Optional[Callable[..., Any]]:
        def transform_func(x):
            return {"input": x}

        return transform_func

    @staticmethod
    def get_reduction_axes() -> int:
        return 0
