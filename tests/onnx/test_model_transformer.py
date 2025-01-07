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

from collections import Counter

import numpy as np
import onnx
import onnxruntime as rt
import pytest

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.onnx_helper import get_tensor
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.graph.transformations.commands import ONNXInitializerUpdateCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.quantization.quantizer_parameters import ONNXQuantizerLayerParameters
from tests.onnx.models import LinearModel
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import min_max_quantize_model

TARGET_LAYERS = [("Non_Existing_Edge",), ("Conv1",), ("Conv1", "BN1", "ReLU1")]
SHOULD_RAISE_EXCEPTION = [True, False, False]
QUANTIZER_NUMBER = [None, 1, 3]


@pytest.mark.parametrize(
    "target_layers, should_raise, quantizer_number", zip(TARGET_LAYERS, SHOULD_RAISE_EXCEPTION, QUANTIZER_NUMBER)
)
def test_quantizer_insertion(target_layers, should_raise, quantizer_number):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()
    nncf_graph = GraphConverter.create_nncf_graph(model)
    nncf_input_node_next_onnx_nodes = {}
    for input_node in nncf_graph.get_input_nodes():
        next_nodes = nncf_graph.get_next_nodes(input_node)
        nncf_input_node_next_onnx_nodes[input_node.node_name] = [node.node_name for node in next_nodes]
    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, target_layer, 0)
        command = ONNXQuantizerInsertionCommand(
            target_point,
            nncf_input_node_next_onnx_nodes,
            ONNXQuantizerLayerParameters(np.array(1), np.array(0), tensor_type=np.uint8),
        )
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)
    if should_raise:
        try:
            _ = model_transformer.transform(transformation_layout)
        except KeyError:
            return
    transformed_model = model_transformer.transform(transformation_layout)
    onnx.checker.check_model(transformed_model)

    num_q = 0
    num_dq = 0

    for node in transformed_model.graph.node:
        op_type = node.op_type
        if op_type == "QuantizeLinear":
            num_q += 1
        elif op_type == "DequantizeLinear":
            num_dq += 1
    assert num_q == num_dq == quantizer_number


TARGET_LAYERS = ["Conv1", "BN1", "ReLU1"]
QUANTIZER_SCALES = [np.array(3.0), 13.2 * np.ones((32)), np.array(17.1)]
QUANTIZER_ZERO_POINT = [np.array(1, dtype=np.int32), 2 * np.ones((32), dtype=np.int32), np.array(0, dtype=np.int32)]
QUANTIZER_ONNX_DTYPE = [np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.uint8)]
QUANTIZER_ONNX_ATTRIBUTES = [{"axis": 0}, {"axis": 0}, {"axis": 0}]


class QuantizerParameters:
    def __init__(self, target_layer, scale, zero_point, onnx_dtype, onnx_attributes):
        self.target_layer = target_layer
        self.scale = scale
        self.zero_point = zero_point
        self.onnx_dtype = onnx_dtype
        self.onnx_attributes = onnx_attributes


@pytest.mark.parametrize(
    "test_parameters",
    [
        QuantizerParameters(*attrs)
        for attrs in zip(
            TARGET_LAYERS, QUANTIZER_SCALES, QUANTIZER_ZERO_POINT, QUANTIZER_ONNX_DTYPE, QUANTIZER_ONNX_ATTRIBUTES
        )
    ],
)
def test_inserted_quantizer_parameters(test_parameters):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()
    quantizer_parameters = ONNXQuantizerLayerParameters(
        test_parameters.scale, test_parameters.zero_point, tensor_type=test_parameters.onnx_dtype
    )
    target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, test_parameters.target_layer, 0)

    nncf_graph = GraphConverter.create_nncf_graph(model)
    nncf_input_node_next_onnx_nodes = {}
    for input_node in nncf_graph.get_input_nodes():
        next_nodes = nncf_graph.get_next_nodes(input_node)
        nncf_input_node_next_onnx_nodes[input_node.node_name] = [node.node_name for node in next_nodes]

    command = ONNXQuantizerInsertionCommand(target_point, nncf_input_node_next_onnx_nodes, quantizer_parameters)
    transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    onnx.checker.check_model(transformed_model)

    for node in transformed_model.graph.node:
        op_type = node.op_type
        if op_type == "QuantizeLinear":
            for attr in node.attribute:
                assert test_parameters.onnx_attributes[attr.name] == onnx.helper.get_attribute_value(attr)
            assert np.allclose(get_tensor_value(transformed_model, node.input[1]), np.array(test_parameters.scale))
            assert np.allclose(get_tensor_value(transformed_model, node.input[2]), np.array(test_parameters.zero_point))
            assert get_tensor_value(transformed_model, node.input[2]).dtype == test_parameters.onnx_dtype


TARGET_LAYERS = [["ReLU1"], ["Conv1", "BN1"], ["Conv1", "BN1", "ReLU1"]]
TARGET_LAYERS_OUTPUT = [["Y", "ReLU1_Y"], ["Y", "Conv1_Y", "BN1_Y"], ["Y", "Conv1_Y", "BN1_Y", "ReLU1_Y"]]


@pytest.mark.parametrize("target_layers, target_layer_outputs", zip(TARGET_LAYERS, TARGET_LAYERS_OUTPUT))
def test_output_insertion(target_layers, target_layer_outputs):
    model = LinearModel().onnx_model
    nncf_graph = GraphConverter.create_nncf_graph(model)
    nncf_input_node_next_onnx_nodes = {}
    for input_node in nncf_graph.get_input_nodes():
        next_nodes = nncf_graph.get_next_nodes(input_node)
        nncf_input_node_next_onnx_nodes[input_node.node_name] = [node.node_name for node in next_nodes]

    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, target_layer, 0)
        command = ONNXOutputInsertionCommand(target_point, nncf_input_node_next_onnx_nodes)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)

    assert Counter([out.name for out in transformed_model.graph.output]) == Counter(target_layer_outputs)


CONV_LAYERS = [["Conv1", "Conv2"]]
BIAS_VALUES = [[np.full((32,), 2), np.full((10,), 3)]]
BIAS_REFERENCES = [[2.0, 3.0]]


@pytest.mark.parametrize("layers, values, refs", zip(CONV_LAYERS, BIAS_VALUES, BIAS_REFERENCES))
def test_bias_correction(layers, values, refs):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()
    for conv_layer, bias_value in zip(layers, values):
        bias_port_id = 2
        target_point = ONNXTargetPoint(TargetType.LAYER, conv_layer, bias_port_id)
        command = ONNXInitializerUpdateCommand(target_point, bias_value)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    node_dict = {node.name: node for node in transformed_model.graph.node}

    for conv_layer, bias_reference in zip(layers, refs):
        bias_tensor_name = node_dict[conv_layer].input[2]
        bias_tensor = get_tensor(transformed_model, bias_tensor_name)
        bias_value = onnx.numpy_helper.to_array(bias_tensor)
        assert np.all(bias_value == bias_reference)


TARGET_LAYERS = [
    ("DequantizeLinear_X_1", "QuantizeLinear_X_1", "QuantizeLinear_Avg_Pool1_Y_1", "DequantizeLinear_Avg_Pool1_Y_1")
]


@pytest.mark.parametrize("target_layers", TARGET_LAYERS)
def test_node_removing(target_layers):
    model_to_test = LinearModel()
    onnx_model = model_to_test.onnx_model

    quantized_model = min_max_quantize_model(onnx_model)

    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.LAYER, target_layer, 0)
        command = ONNXQDQNodeRemovingCommand(target_point)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(quantized_model)

    transformed_model = model_transformer.transform(transformation_layout)
    onnx.checker.check_model(transformed_model)
    compare_nncf_graph(transformed_model, "synthetic/" + "removed_nodes_in_" + model_to_test.path_ref_graph)


def test_no_transformations():
    def infer_model_with_ones(model, shape):
        model = model.SerializeToString()
        sess = rt.InferenceSession(model, providers=["CPUExecutionProvider"])
        _input = np.ones(shape)
        input_name = sess.get_inputs()[0].name
        return sess.run([], {input_name: _input.astype(np.float32)})

    onnx_model = LinearModel().onnx_model
    input_shape = [1, 3, 32, 32]
    model_transformer = ONNXModelTransformer(onnx_model)
    transformed_model = model_transformer.transform(TransformationLayout())

    ret_val_1 = infer_model_with_ones(onnx_model, input_shape)
    ret_val_2 = infer_model_with_ones(transformed_model, input_shape)
    assert np.allclose(ret_val_1, ret_val_2)
    assert id(transformed_model) != id(onnx_model)
