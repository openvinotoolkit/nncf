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
import numpy as np
import openvino.runtime as ov

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.openvino_native.graph.model_transformer import OVModelTransformer
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino_native.quantization.quantizer_parameters import OVQuantizerLayerParameters
from nncf.experimental.openvino_native.graph.transformations.commands import OVBiasCorrectionCommand

from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import FPModel
from tests.openvino.native.models import QuantizedModel
from tests.openvino.native.common import compare_nncf_graphs
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT

REFERENCE_GRAPHS_DIR = OPENVINO_NATIVE_TEST_ROOT / 'data' / 'reference_graphs' / 'original_nncf_graph'

TARGET_INSERT_LAYERS = [['Add'], ['MatMul'], ['Add', 'MatMul']]
TARGET_PRE_LAYERS_OUTPUT = [['Result_Reshape.0'], ['Result_Reshape.0'], ['Result_Reshape.0']]
TARGET_POST_LAYERS_OUTPUT = [['Result_Add.0'], ['Result_MatMul.0'], ['Result_Add.0', 'Result_MatMul.0']]
TARGET_PRE_LAYER_FQS = [['Add/fq_input_0'], ['MatMul/fq_input_0'], ['Add/fq_input_0', 'MatMul/fq_input_0']]
TARGET_POST_LAYER_FQS = [['Add/fq_output_0'], ['MatMul/fq_output_0'], ['Add/fq_output_0', 'MatMul/fq_output_0']]
TARGET_WEIGHTS_FQS = [['Add/fq_weights_1'], ['MatMul/fq_weights_1'], ['Add/fq_weights_1', 'MatMul/fq_weights_1']]


def create_transformed_model(model, target_layers, target_type, command_type, port_id=0, **kwargs):
    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = OVTargetPoint(target_type, target_layer, port_id=port_id)
        command = command_type(target_point, **kwargs)
        transformation_layout.register(command)

    model_transformer = OVModelTransformer(model)
    transformed_model = model_transformer.transform(transformation_layout)
    return transformed_model


def get_extra_outputs(original_model, transformed_model):
    extra_outputs = set()
    for out in transformed_model.get_results():
        extra_outputs.add(out.get_friendly_name())

    for out in original_model.get_results():
        extra_outputs.remove(out.get_friendly_name())

    return extra_outputs


def get_fq_nodes(model):
    fq_nodes = []
    for op in model.get_ops():
        if op.get_type_name() == 'FakeQuantize':
            fq_nodes.append(op.get_friendly_name())

    return fq_nodes


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_INSERT_LAYERS, TARGET_PRE_LAYERS_OUTPUT))
def test_output_insertion_pre_layer(target_layers, target_layer_outputs):
    model = LinearModel().ov_model
    transformed_model = create_transformed_model(
        model, target_layers, TargetType.PRE_LAYER_OPERATION, OVOutputInsertionCommand)
    extra_outputs = get_extra_outputs(model, transformed_model)

    assert len(extra_outputs) == len(target_layer_outputs)
    for out_name in extra_outputs:
        assert out_name in target_layer_outputs


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_INSERT_LAYERS, TARGET_POST_LAYERS_OUTPUT))
def test_output_insertion_post_layer(target_layers, target_layer_outputs):
    model = LinearModel().ov_model
    transformed_model = create_transformed_model(
        model, target_layers, TargetType.POST_LAYER_OPERATION, OVOutputInsertionCommand)
    extra_outputs = get_extra_outputs(model, transformed_model)

    assert len(extra_outputs) == len(target_layer_outputs)
    for out_name in extra_outputs:
        assert out_name in target_layer_outputs


TARGET_LAYERS = [('Conv_1/fq_input_0', 'Concat_1/fq_input_0', 'Conv_3/fq_weights_0', 'Add_2/fq_weights_0')]

@pytest.mark.parametrize('target_layers', TARGET_LAYERS)
def test_node_removing(target_layers):
    model_to_test = QuantizedModel()
    model = model_to_test.ov_model

    transformation_layout = TransformationLayout()

    for target_layer in target_layers:
        target_point = OVTargetPoint(TargetType.LAYER, target_layer, 0)
        command = OVFQNodeRemovingCommand(target_point)
        transformation_layout.register(command)

    model_transformer = OVModelTransformer(model)

    transformed_model = model_transformer.transform(transformation_layout)
    ref_name = 'removed_nodes_in_' + model_to_test.ref_graph_name
    compare_nncf_graphs(transformed_model, REFERENCE_GRAPHS_DIR / ref_name)


@pytest.mark.parametrize('target_layers, ref_fq_names', zip(TARGET_INSERT_LAYERS, TARGET_PRE_LAYER_FQS))
def test_fq_insertion_pre_layer(target_layers, ref_fq_names):
    model = LinearModel().ov_model

    min_values = np.zeros((1, 1, 1, 1)).astype(np.float32)
    max_values = np.ones((1, 1, 1, 1)).astype(np.float32)
    quantizer_parameters = OVQuantizerLayerParameters(min_values, max_values, min_values, max_values, levels=256)

    transformed_model = create_transformed_model(model, target_layers, TargetType.PRE_LAYER_OPERATION,
            OVQuantizerInsertionCommand, quantizer_parameters=quantizer_parameters)
    fq_nodes = get_fq_nodes(transformed_model)

    assert len(fq_nodes) == len(ref_fq_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fq_names


@pytest.mark.parametrize('target_layers, ref_fq_names', zip(TARGET_INSERT_LAYERS, TARGET_POST_LAYER_FQS))
def test_fq_insertion_post_layer(target_layers, ref_fq_names):
    model = LinearModel().ov_model

    min_values = np.zeros((1, 1, 1, 1)).astype(np.float32)
    max_values = np.ones((1, 1, 1, 1)).astype(np.float32)
    quantizer_parameters = OVQuantizerLayerParameters(min_values, max_values, min_values, max_values, levels=256)
    transformed_model = create_transformed_model(model, target_layers, TargetType.POST_LAYER_OPERATION,
            OVQuantizerInsertionCommand, quantizer_parameters=quantizer_parameters)
    fq_nodes = get_fq_nodes(transformed_model)

    assert len(fq_nodes) == len(ref_fq_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fq_names


@pytest.mark.parametrize('target_layers, ref_fq_names', zip(TARGET_INSERT_LAYERS, TARGET_WEIGHTS_FQS))
def test_fq_insertion_weights(target_layers, ref_fq_names):
    model = LinearModel().ov_model

    min_values = np.zeros((1, 1, 1, 1)).astype(np.float32)
    max_values = np.ones((1, 1, 1, 1)).astype(np.float32)
    quantizer_parameters = OVQuantizerLayerParameters(min_values, max_values, min_values, max_values, levels=256)
    transformed_model = create_transformed_model(model, target_layers, TargetType.OPERATION_WITH_WEIGHTS,
            OVQuantizerInsertionCommand, port_id=1, quantizer_parameters=quantizer_parameters)
    fq_nodes = get_fq_nodes(transformed_model)

    assert len(fq_nodes) == len(ref_fq_names)
    for fq_name in fq_nodes:
        assert fq_name in ref_fq_names


MODELS_WITH_PARAMETERS = [
    {
        'model': ConvModel().ov_model,
        'layers': ['Conv'],
        'values': [np.full((3,), 2)],
        'refs': [2.0],
    },
    {
        'model': FPModel(precision='FP16').ov_model,
        'layers': ['MatMul'],
        'values': [np.full((3,), 2)],
        'refs': [2.0],
    }
]


@pytest.mark.parametrize('model_with_parameters', MODELS_WITH_PARAMETERS)
def test_bias_correction(model_with_parameters):
    model = model_with_parameters['model']
    layers = model_with_parameters['layers']
    values = model_with_parameters['values']
    refs = model_with_parameters['refs']

    transformed_model = create_transformed_model(model, layers, TargetType.LAYER,
                                                 OVBiasCorrectionCommand, port_id=1, **{'bias_value': values})
    ops_dict = {op.get_friendly_name(): op for op in transformed_model.get_ops()}

    for node_name, bias_reference in zip(layers, refs):
        node = ops_dict[node_name]
        node_inputs = [port.get_node() for port in node.output(0).get_target_inputs()]
        node_with_bias = node_inputs[0]

        potential_bias = node_with_bias.input_value(1).node
        if potential_bias.get_type_name() == 'Convert':
            potential_bias = potential_bias.input_value(0).node
        assert potential_bias.get_type_name() == 'Constant'
        assert np.all(potential_bias.get_vector() == bias_reference)


def test_no_transformations():
    def infer_model_with_ones(model, shape):
        ie = ov.Core()
        compiled_model = ie.compile_model(model)
        _input = np.ones(shape)
        model_outputs = compiled_model(_input)
        return {out.get_node().get_friendly_name(): data for out, data in model_outputs.items()}

    model = LinearModel().ov_model
    input_shape = [1, 3, 4, 2]
    model_transformer = OVModelTransformer(model)
    transformed_model = model_transformer.transform(TransformationLayout())

    ret_val_1 = infer_model_with_ones(model, input_shape)
    ret_val_2 = infer_model_with_ones(transformed_model, input_shape)
    assert ret_val_1.keys() == ret_val_2.keys()
    for output in ret_val_1.keys():
        assert np.allclose(ret_val_1[output], ret_val_2[output])
    assert id(transformed_model) != id(model)
