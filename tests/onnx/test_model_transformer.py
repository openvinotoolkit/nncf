"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pytest

import onnx
# pylint: disable=no-member
import numpy as np
from nncf.common.graph.transformations.layout import TransformationLayout

from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.common.graph.transformations.commands import TargetType
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXQDQNodeRemovingCommand
from nncf.common.quantization.structs import QuantizationMode
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.quantization.quantizer_parameters import ONNXQuantizerLayerParameters

from tests.onnx.models import LinearModel
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph

TARGET_LAYERS = [('Non_Existing_Edge',), ('Conv1',), ('Conv1', 'BN1', 'ReLU1')]
SHOULD_RAISE_EXCEPTION = [True, False, False]
QUANTIZER_NUMBER = [None, 1, 3]


@pytest.mark.parametrize("target_layers, should_raise, quantizer_number",
                         zip(TARGET_LAYERS, SHOULD_RAISE_EXCEPTION, QUANTIZER_NUMBER))
def test_quantizer_insertion(target_layers, should_raise, quantizer_number):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()

    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, target_layer, 0)
        command = ONNXQuantizerInsertionCommand(
            target_point,
            ONNXQuantizerLayerParameters(np.array(1.0), np.array(0),
                                         QuantizationMode.SYMMETRIC, None,
                                         tensor_type=np.int8))
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)
    if should_raise:
        try:
            _ = model_transformer.transform(transformation_layout)
        except RuntimeError:
            return
    transformed_model = model_transformer.transform(transformation_layout)
    onnx.checker.check_model(transformed_model)

    num_q = 0
    num_dq = 0
    # pylint:disable=no-member
    for node in transformed_model.graph.node:
        op_type = node.op_type
        if op_type == 'QuantizeLinear':
            num_q += 1
        elif op_type == 'DequantizeLinear':
            num_dq += 1
    assert num_q == num_dq == quantizer_number


TARGET_LAYERS = ['Conv1', 'BN1', 'ReLU1']
QUANTIZER_SCALES = [np.array(3.0), 13.2 * np.ones((32)), np.array(17.1)]
QUANTIZER_ZERO_POINT = [np.array(1, dtype=np.int32), 2 * np.ones((32), dtype=np.int32), np.array(0, dtype=np.int32)]
QUANTIZER_MODE = [QuantizationMode.SYMMETRIC, QuantizationMode.SYMMETRIC, QuantizationMode.ASYMMETRIC]
QUANTIZER_ONNX_DTYPE = [np.dtype(np.int8), np.dtype(np.int8), np.dtype(np.uint8)]
QUANTIZER_ONNX_ATTRIBUTES = [{'axis': 0}, {'axis': 0}, {'axis': 0}]


class QuantizerParameters:
    def __init__(self, target_layer, scale, zero_point, mode, onnx_dtype, onnx_attributes):
        self.target_layer = target_layer
        self.scale = scale
        self.zero_point = zero_point
        self.mode = mode
        self.onnx_dtype = onnx_dtype
        self.onnx_attributes = onnx_attributes


@pytest.mark.parametrize("test_parameters", [QuantizerParameters(*attrs) for attrs in
                                             zip(TARGET_LAYERS, QUANTIZER_SCALES, QUANTIZER_ZERO_POINT,
                                                 QUANTIZER_MODE, QUANTIZER_ONNX_DTYPE, QUANTIZER_ONNX_ATTRIBUTES)])
def test_inserted_quantizer_parameters(test_parameters):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()
    quantizer_parameters = ONNXQuantizerLayerParameters(test_parameters.scale, test_parameters.zero_point,
                                                        test_parameters.mode, None,
                                                        tensor_type=test_parameters.onnx_dtype)
    target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, test_parameters.target_layer, 0)
    command = ONNXQuantizerInsertionCommand(target_point, quantizer_parameters)
    transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    onnx.checker.check_model(transformed_model)

    onnx_graph = ONNXGraph(transformed_model)

    # pylint:disable=no-member
    for node in transformed_model.graph.node:
        op_type = node.op_type
        if op_type == 'QuantizeLinear':
            for attr in node.attribute:
                assert test_parameters.onnx_attributes[attr.name] == onnx.helper.get_attribute_value(attr)
            assert np.allclose(onnx_graph.get_initializers_value(node.input[1]), np.array(test_parameters.scale))
            assert np.allclose(onnx_graph.get_initializers_value(node.input[2]), np.array(test_parameters.zero_point))
            assert onnx_graph.get_initializers_value(node.input[2]).dtype == test_parameters.onnx_dtype


TARGET_LAYERS = [['ReLU1'], ['Conv1', 'BN1'], ['Conv1', 'BN1', 'ReLU1']]
TARGET_LAYERS_OUTPUT = [['ReLU1_Y'], ['Conv1_Y', 'BN1_Y'], ['Conv1_Y', 'BN1_Y', 'ReLU1_Y']]


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_LAYERS, TARGET_LAYERS_OUTPUT))
def test_output_insertion(target_layers, target_layer_outputs):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.POST_LAYER_OPERATION, target_layer, 0)
        command = ONNXOutputInsertionCommand(target_point)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    # TODO(kshpv): The problem occurs because shaope field is missing,
    #  but this is essential for some dynamic models such as Mask-RCNN
    # onnx.checker.check_model(transformed_model)

    onnx_graph = ONNXGraph(transformed_model)
    # Should be topologically sorted
    for i in range(len(target_layers)):
        assert onnx_graph.get_model_outputs()[i].name in target_layer_outputs


CONV_LAYERS = [['Conv1', 'Conv2']]
BIAS_VALUES = [[np.full((32,), 2), np.full((10,), 3)]]
BIAS_REFERENCES = [[2.0, 3.0]]


@pytest.mark.parametrize('layers, values, refs', zip(CONV_LAYERS, BIAS_VALUES, BIAS_REFERENCES))
def test_bias_correction(layers, values, refs):
    model = LinearModel().onnx_model
    transformation_layout = TransformationLayout()
    for conv_layer, bias_value in zip(layers, values):
        bias_port_id = 2
        target_point = ONNXTargetPoint(TargetType.LAYER, conv_layer, bias_port_id)
        command = ONNXBiasCorrectionCommand(target_point, bias_value)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    onnx_graph = ONNXGraph(transformed_model)

    for conv_layer, bias_reference in zip(layers, refs):
        bias_tensor_name = onnx_graph.get_node_by_name(conv_layer).input[2]
        bias_tensor = onnx_graph.get_initializer(bias_tensor_name)
        bias_value = onnx.numpy_helper.to_array(bias_tensor)
        assert np.all(bias_value == bias_reference)


TARGET_LAYERS = [('DequantizeLinear_X_1',
                  'QuantizeLinear_X_1',
                  'QuantizeLinear_Avg_Pool1_Y_1',
                  'DequantizeLinear_Avg_Pool1_Y_1')]

@pytest.mark.parametrize('target_layers', TARGET_LAYERS)
def test_node_removing(target_layers):
    model_to_test = LinearModel()
    onnx_model = model_to_test.onnx_model

    quantized_model = min_max_quantize_model(model_to_test.input_shape[0], onnx_model)

    transformation_layout = TransformationLayout()

    for target_layer in target_layers:
        target_point = ONNXTargetPoint(TargetType.LAYER, target_layer, 0)
        command = ONNXQDQNodeRemovingCommand(target_point)
        transformation_layout.register(command)

    model_transformer = ONNXModelTransformer(quantized_model)

    transformed_model = model_transformer.transform(transformation_layout)
    compare_nncf_graph(transformed_model, 'synthetic/' + 'removed_nodes_in_' + model_to_test.path_ref_graph)
