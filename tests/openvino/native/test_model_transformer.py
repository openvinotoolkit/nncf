"""
 Copyright (c) 2022 Intel Corporation
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

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.experimental.openvino_native.engine import OVNativeEngine
from nncf.experimental.openvino_native.graph.model_transformer import OVModelTransformer
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand

from tests.openvino.native.models import LinearModel

REF_OUTPUT_SHAPES = {'Result_MatMul': (1, 3, 2, 5), 'Result_Add': (1, 3, 2, 4)}
TARGET_LAYERS = [['Add'], ['MatMul'], ['Add', 'MatMul']]
TARGET_PRE_LAYERS_OUTPUT = [['Result_Reshape.0'], ['Result_Reshape.0'], ['Result_Reshape.0']]
TARGET_POST_LAYERS_OUTPUT = [['Result_Add.0'], ['Result_MatMul.0'], ['Result_Add.0', 'Result_MatMul.0']]


def test_infer_original_model():
    model = LinearModel().ov_model
    input_data = {inp.get_friendly_name(): np.random.rand(*inp.shape) for inp in model.get_parameters()}

    engine = OVNativeEngine(model)
    outputs = engine.infer(input_data)
    for out_name, out in outputs.items():
        assert out.shape == REF_OUTPUT_SHAPES[out_name]


def create_transformed_model(model, target_layers, target_type, command_type):
    transformation_layout = TransformationLayout()
    for target_layer in target_layers:
        target_point = OVTargetPoint(target_type, target_layer, port_id=0)
        command = command_type(target_point)
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


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_LAYERS, TARGET_PRE_LAYERS_OUTPUT))
def test_output_insertion_pre_layer(target_layers, target_layer_outputs):
    model = LinearModel().ov_model
    transformed_model = create_transformed_model(
        model, target_layers, TargetType.PRE_LAYER_OPERATION, OVOutputInsertionCommand)
    extra_outputs = get_extra_outputs(model, transformed_model)

    assert len(extra_outputs) == len(target_layer_outputs)
    for out_name in extra_outputs:
        assert out_name in target_layer_outputs


@pytest.mark.parametrize('target_layers, target_layer_outputs', zip(TARGET_LAYERS, TARGET_POST_LAYERS_OUTPUT))
def test_output_insertion_post_layer(target_layers, target_layer_outputs):
    model = LinearModel().ov_model
    transformed_model = create_transformed_model(
        model, target_layers, TargetType.POST_LAYER_OPERATION, OVOutputInsertionCommand)
    extra_outputs = get_extra_outputs(model, transformed_model)

    assert len(extra_outputs) == len(target_layer_outputs)
    for out_name in extra_outputs:
        assert out_name in target_layer_outputs
