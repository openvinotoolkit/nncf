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

import os

import pytest
import tensorflow as tf
from packaging import version
from tensorflow.keras import layers
from tensorflow.keras import models

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.graph.transformations.commands import TransformationType
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.dot_file_rw import write_dot_graph
from nncf.tensorflow import tf_internals
from nncf.tensorflow.graph.converter import TFModelConverterFactory
from nncf.tensorflow.graph.model_transformer import TFModelTransformer
from nncf.tensorflow.graph.transformations import commands
from nncf.tensorflow.graph.transformations.commands import TFInsertionCommand
from nncf.tensorflow.graph.transformations.commands import TFLayer
from nncf.tensorflow.graph.transformations.commands import TFLayerWeight
from nncf.tensorflow.graph.transformations.commands import TFMultiLayerPoint
from nncf.tensorflow.graph.transformations.commands import TFMultipleInsertionCommands
from nncf.tensorflow.graph.transformations.layout import TFTransformationLayout
from nncf.tensorflow.graph.utils import is_functional_model
from nncf.tensorflow.layers.custom_objects import NNCF_CUSTOM_OBJECTS
from nncf.tensorflow.layers.operation import NNCFOperation
from nncf.tensorflow.quantization.layers import FakeQuantize
from nncf.tensorflow.quantization.quantizers import TFQuantizerSpec
from tests.cross_fw.shared.nx_graph import compare_nx_graph_with_reference
from tests.tensorflow.helpers import remove_node_by_name
from tests.tensorflow.test_compressed_graph import get_nx_graph_from_tf_graph
from tests.tensorflow.test_compressed_graph import keras_model_to_tf_graph


def test_insertion_commands_union_invalid_input():
    cmd_0 = commands.TFInsertionCommand(commands.TFBeforeLayer("layer_0"))
    cmd_1 = commands.TFInsertionCommand(commands.TFAfterLayer("layer_0"))
    with pytest.raises(Exception):
        cmd_0.union(cmd_1)


priority_types = ["same", "different"]


@pytest.mark.parametrize("case", priority_types, ids=priority_types)
def test_insertion_command_priority(case):
    def make_operation_fn(priority_value):
        def operation_fn():
            return priority_value

        return operation_fn

    cmds = []
    if case == "same":
        for idx in range(3):
            cmds.append(commands.TFInsertionCommand(commands.TFBeforeLayer("layer_0"), make_operation_fn(idx)))
    else:
        priorites = sorted(list(TransformationPriority), key=lambda x: x.value, reverse=True)
        for priority in priorites:
            cmds.append(
                commands.TFInsertionCommand(
                    commands.TFLayerWeight("layer_0", "weight_0"), make_operation_fn(priority.value), priority
                )
            )

    res_cmd = cmds[0]
    for cmd in cmds[1:]:
        res_cmd = res_cmd + cmd

    res = res_cmd.insertion_objects
    assert len(res) == len(cmds)
    assert all(res[i]() <= res[i + 1]() for i in range(len(res) - 1))


def test_insertion_commands_union_multi_layer_point():
    cmd_1 = TFMultiLayerPoint(
        [
            commands.TFAfterLayer("layer_0"),
            commands.TFBeforeLayer("layer_1"),
        ]
    )
    cmd_2 = TFMultiLayerPoint([commands.TFBeforeLayer("layer_2"), commands.TFAfterLayer("layer_3")])
    cmd_0 = commands.TFInsertionCommand(cmd_1)
    cmd_1 = commands.TFInsertionCommand(cmd_2)
    with pytest.raises(Exception):
        cmd_0.union(cmd_1)


def test_removal_command_union():
    cmd_0 = commands.TFRemovalCommand(commands.TFLayer("layer_0"))
    cmd_1 = commands.TFRemovalCommand(commands.TFLayer("layer_1"))
    with pytest.raises(Exception):
        cmd_0.union(cmd_1)


def test_add_insertion_command_to_multiple_insertion_commands_same():
    check_fn = lambda src, dst: dst.type == TargetType.OPERATION_WITH_WEIGHTS and src.layer_name == dst.layer_name

    cmd_0 = commands.TFInsertionCommand(commands.TFLayerWeight("layer_0", "weight_0"), lambda: "cmd_0")

    cmd_1 = commands.TFInsertionCommand(commands.TFLayerWeight("layer_0", "weight_0"), lambda: "cmd_1")

    m_cmd = commands.TFMultipleInsertionCommands(
        target_point=commands.TFLayer("layer_0"), check_target_points_fn=check_fn
    )

    m_cmd.add_insertion_command(cmd_0)
    m_cmd.add_insertion_command(cmd_1)

    res_cmds = m_cmd.commands
    assert len(res_cmds) == 1

    res = res_cmds[0].insertion_objects
    assert len(res) == 2
    assert res[0]() == "cmd_0"
    assert res[1]() == "cmd_1"


def test_add_insertion_command_to_multiple_insertion_commands_different():
    check_fn = lambda src, dst: dst.type == TargetType.OPERATION_WITH_WEIGHTS and src.layer_name == dst.layer_name

    cmd_0 = commands.TFInsertionCommand(commands.TFLayerWeight("layer_0", "weight_0"), lambda: "cmd_0")

    cmd_1 = commands.TFInsertionCommand(commands.TFLayerWeight("layer_0", "weight_1"), lambda: "cmd_1")

    m_cmd = commands.TFMultipleInsertionCommands(
        target_point=commands.TFLayer("layer_0"), check_target_points_fn=check_fn
    )

    m_cmd.add_insertion_command(cmd_0)
    m_cmd.add_insertion_command(cmd_1)

    res_cmds = m_cmd.commands
    assert len(res_cmds) == 2

    res = res_cmds[0].insertion_objects
    assert len(res) == 1
    assert res[0]() == "cmd_0"

    res = res_cmds[1].insertion_objects
    assert len(res) == 1
    assert res[0]() == "cmd_1"


def test_add_insertion_command_to_multiple_insertion_commands_invalid_input():
    m_cmd = commands.TFMultipleInsertionCommands(commands.TFLayerWeight("layer_0", "weights_0"))
    cmd = commands.TFRemovalCommand(commands.TFLayer("layer_0"))
    with pytest.raises(Exception):
        m_cmd.add_insertion_command(cmd)


def test_multiple_insertion_commands_union_invalid_input():
    cmd_0 = commands.TFMultipleInsertionCommands(commands.TFLayer("layer_0"))
    cmd_1 = commands.TFMultipleInsertionCommands(commands.TFLayer("layer_1"))
    with pytest.raises(Exception):
        cmd_0.add_insertion_command(cmd_1)


def test_multiple_insertion_commands_union():
    check_fn_0 = (
        lambda src, dst: dst.type == TargetType.OPERATION_WITH_WEIGHTS
        and src.layer_name == dst.layer_name
        and dst.weights_attr_name == "weight_0"
    )

    cmd_0 = commands.TFInsertionCommand(commands.TFLayerWeight("layer_0", "weight_0"), lambda: "cmd_0")

    m_cmd_0 = commands.TFMultipleInsertionCommands(
        target_point=commands.TFLayer("layer_0"), check_target_points_fn=check_fn_0, commands=[cmd_0]
    )

    check_fn_1 = (
        lambda src, dst: dst.type == TargetType.OPERATION_WITH_WEIGHTS
        and src.layer_name == dst.layer_name
        and dst.weights_attr_name == "weight_1"
    )

    cmd_1 = commands.TFInsertionCommand(commands.TFLayerWeight("layer_0", "weight_1"), lambda: "cmd_1")

    m_cmd_1 = commands.TFMultipleInsertionCommands(
        target_point=commands.TFLayer("layer_0"), check_target_points_fn=check_fn_1, commands=[cmd_1]
    )

    m_cmd = m_cmd_0 + m_cmd_1

    res_cmds = m_cmd.commands
    assert len(res_cmds) == 2

    res = res_cmds[0].insertion_objects
    assert len(res) == 1
    assert res[0]() == "cmd_0"

    res = res_cmds[1].insertion_objects
    assert len(res) == 1
    assert res[0]() == "cmd_1"


def test_transformation_layout_insertion_case():
    transformation_layout = TFTransformationLayout()

    check_fn = lambda src, dst: dst.type == TargetType.OPERATION_WITH_WEIGHTS and src.layer_name == dst.layer_name

    command_list = [
        commands.TFInsertionCommand(
            commands.TFLayerWeight("layer_0", "weight_0"),
            lambda: "cmd_0",
            TransformationPriority.SPARSIFICATION_PRIORITY,
        ),
        commands.TFInsertionCommand(
            commands.TFLayerWeight("layer_0", "weight_1"),
            lambda: "cmd_1",
            TransformationPriority.SPARSIFICATION_PRIORITY,
        ),
        commands.TFInsertionCommand(
            commands.TFLayerWeight("layer_1", "weight_0"),
            lambda: "cmd_2",
            TransformationPriority.SPARSIFICATION_PRIORITY,
        ),
        commands.TFMultipleInsertionCommands(
            target_point=commands.TFLayer("layer_0"),
            check_target_points_fn=check_fn,
            commands=[
                commands.TFInsertionCommand(
                    commands.TFLayerWeight("layer_0", "weight_0"),
                    lambda: "cmd_3",
                    TransformationPriority.PRUNING_PRIORITY,
                )
            ],
        ),
        commands.TFMultipleInsertionCommands(
            target_point=commands.TFLayer("layer_1"),
            check_target_points_fn=check_fn,
            commands=[
                commands.TFInsertionCommand(
                    commands.TFLayerWeight("layer_1", "weight_0"),
                    lambda: "cmd_4",
                    TransformationPriority.PRUNING_PRIORITY,
                ),
                commands.TFInsertionCommand(
                    commands.TFLayerWeight("layer_1", "weight_1"),
                    lambda: "cmd_5",
                    TransformationPriority.PRUNING_PRIORITY,
                ),
            ],
        ),
    ]

    for cmd in command_list:
        transformation_layout.register(cmd)

    res_transformations = transformation_layout.transformations
    assert len(res_transformations) == 2
    assert res_transformations[0].type == TransformationType.MULTI_INSERT
    assert res_transformations[0].target_point.type == TargetType.LAYER
    assert res_transformations[0].target_point.layer_name == "layer_0"
    assert res_transformations[1].type == TransformationType.MULTI_INSERT
    assert res_transformations[1].target_point.type == TargetType.LAYER
    assert res_transformations[1].target_point.layer_name == "layer_1"

    res_cmds = res_transformations[0].commands
    assert len(res_cmds) == 2

    res = res_cmds[0].insertion_objects
    assert len(res) == 2
    assert res[0]() == "cmd_3" and res[1]() == "cmd_0"

    res = res_cmds[1].insertion_objects
    assert len(res) == 1
    assert res[0]() == "cmd_1"

    res_cmds = res_transformations[1].commands
    assert len(res_cmds) == 2

    res = res_cmds[0].insertion_objects
    assert len(res) == 2
    assert res[0]() == "cmd_4" and res[1]() == "cmd_2"

    res = res_cmds[1].insertion_objects
    assert len(res) == 1
    assert res[0]() == "cmd_5"


def test_transformation_layout_removal_case():
    transformation_layout = TFTransformationLayout()

    command_list = [
        commands.TFInsertionCommand(
            commands.TFLayerWeight("layer_0", "weight_0"),
            lambda: "sparsity_operation",
            TransformationPriority.SPARSIFICATION_PRIORITY,
        ),
        commands.TFRemovalCommand(commands.TFOperationWithWeights("layer_0", "weight_0", "sparsity_operation")),
        commands.TFInsertionCommand(commands.TFAfterLayer("layer_0"), lambda: "layer_1"),
        commands.TFRemovalCommand(commands.TFLayer("layer_1")),
        commands.TFInsertionCommand(
            commands.TFLayerWeight("layer_0", "weight_0"),
            lambda: "pruning_operation",
            TransformationPriority.PRUNING_PRIORITY,
        ),
    ]

    for cmd in command_list:
        transformation_layout.register(cmd)

    res_transformations = transformation_layout.transformations
    assert len(res_transformations) == 5
    assert res_transformations[0].type == TransformationType.INSERT
    assert res_transformations[0].target_point.type == TargetType.OPERATION_WITH_WEIGHTS
    assert res_transformations[0].target_point.layer_name == "layer_0"
    assert res_transformations[0].target_point.weights_attr_name == "weight_0"

    assert res_transformations[1].type == TransformationType.REMOVE
    assert res_transformations[1].target_point.type == TargetType.OPERATION_WITH_WEIGHTS
    assert res_transformations[1].target_point.layer_name == "layer_0"
    assert res_transformations[1].target_point.weights_attr_name == "weight_0"
    assert res_transformations[1].target_point.operation_name == "sparsity_operation"

    assert res_transformations[2].type == TransformationType.INSERT
    assert res_transformations[2].target_point.type == TargetType.AFTER_LAYER
    assert res_transformations[2].target_point.layer_name == "layer_0"

    assert res_transformations[3].type == TransformationType.REMOVE
    assert res_transformations[3].target_point.type == TargetType.LAYER
    assert res_transformations[3].target_point.layer_name == "layer_1"

    assert res_transformations[4].type == TransformationType.INSERT
    assert res_transformations[4].target_point.type == TargetType.OPERATION_WITH_WEIGHTS
    assert res_transformations[4].target_point.layer_name == "layer_0"
    assert res_transformations[4].target_point.weights_attr_name == "weight_0"


CUSTOM_LAYER_NAME = "custom_layer_for_test"


class TwoWeightCustomLayerForTest(tf.keras.layers.Layer):
    WEIGHT_1_NAME = "w1"
    WEIGHT_2_NAME = "w2"

    def __init__(self, name=CUSTOM_LAYER_NAME, trainable=True, dtype="float32"):
        super().__init__(name=name, trainable=trainable, dtype=dtype)
        self.w1 = self.add_weight(shape=(3, 1, 1, 3), name=self.WEIGHT_1_NAME)
        self.w2 = self.add_weight(shape=(3, 1, 1, 3), name=self.WEIGHT_2_NAME)

    def call(self, inputs, **kwargs):
        x = tf.nn.conv2d(inputs, self.w1, strides=[1, 1, 1, 1], padding="SAME")
        x = tf.nn.conv2d(x, self.w2, strides=[1, 1, 1, 1], padding="SAME")
        return x


def ModelWithTwoWeightCustomLayer():
    input_shape = (None, None, 3)
    img_input = layers.Input(name="input", shape=input_shape)
    x = img_input
    x = TwoWeightCustomLayerForTest()(x)  # custom!
    model = models.Model(img_input, x, name="ModelForCustomLayerTest")
    model.build([16, 16, 3])
    return model


def create_transformed_model(transformation_layout: TFTransformationLayout):
    model = ModelWithTwoWeightCustomLayer()
    transformer = TFModelTransformer(model)
    model = transformer.transform(transformation_layout)
    return model


@NNCF_CUSTOM_OBJECTS.register()
class MockIdentityOp(NNCFOperation):
    def build(self, input_shape, input_type, name, layer):
        return {}

    def call(self, inputs, weights, _):
        return inputs


def test_multiple_insertion_command_has_same_effect_as_multiple_single_insertions():
    check_fn = lambda src, dst: dst.type == TargetType.OPERATION_WITH_WEIGHTS

    insertion_command_1 = TFInsertionCommand(
        TFLayerWeight(CUSTOM_LAYER_NAME, TwoWeightCustomLayerForTest.WEIGHT_1_NAME),
        MockIdentityOp("mock_nncf_op_1"),
        TransformationPriority.PRUNING_PRIORITY,
    )
    insertion_command_2 = TFInsertionCommand(
        TFLayerWeight(CUSTOM_LAYER_NAME, TwoWeightCustomLayerForTest.WEIGHT_2_NAME),
        MockIdentityOp("mock_nncf_op_2"),
        TransformationPriority.PRUNING_PRIORITY,
    )
    multiple_insertion_command = TFMultipleInsertionCommands(
        target_point=TFLayer(CUSTOM_LAYER_NAME),
        commands=[insertion_command_1, insertion_command_2],
        check_target_points_fn=check_fn,
    )

    transformation_layout_multi = TFTransformationLayout()
    transformation_layout_multi.register(multiple_insertion_command)
    transformation_layout_two_single = TFTransformationLayout()
    transformation_layout_two_single.register(insertion_command_1)
    transformation_layout_two_single.register(insertion_command_2)

    model_with_multi = create_transformed_model(transformation_layout_multi)
    model_with_two_single = create_transformed_model(transformation_layout_two_single)

    multi_config = model_with_multi.get_config()
    two_single_config = model_with_two_single.get_config()
    assert multi_config == two_single_config


def create_functional_model():
    img_input = layers.Input(name="input", shape=(None, None, 3), dtype="float32")
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding="same", activation="relu")(img_input)

    residual = layers.Conv2D(filters=64, kernel_size=(1, 1), strides=2)(x)
    residual = layers.BatchNormalization()(residual)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([residual, x])

    x = layers.Dense(units=10, activation="softmax")(x)
    model = models.Model(img_input, x, name="ResnetBlockTest")
    return model


def create_sequential_model():
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(None, None, 3)))
    model.add(layers.Conv2D(filters=64, kernel_size=(1, 1), strides=2, activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(2, activation="relu"))
    return model


def apply_insert_after(model):
    converter = TFModelConverterFactory.create(model)
    transformations = TFTransformationLayout()
    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False)

    functional_model = is_functional_model(model)
    for i, layer in enumerate(model.layers):
        original_node_name = layer.name

        if functional_model:
            _, layer_info = converter.get_layer_info_for_node(original_node_name)
            instance_idx = layer_info.instance_idx
        else:
            instance_idx = 0

        fake_quantize_name = f"FakeQuantize_{i}/{original_node_name}"
        fake_quantize_layer = FakeQuantize(
            TFQuantizerSpec.from_config(qconfig, narrow_range=False, half_range=False), name=fake_quantize_name
        )

        transformations.register(
            TFInsertionCommand(
                target_point=commands.TFAfterLayer(original_node_name, instance_idx=instance_idx, output_port_id=0),
                callable_object=fake_quantize_layer,
                priority=TransformationPriority.QUANTIZATION_PRIORITY,
            )
        )

    transformer = TFModelTransformer(model)
    transformed_model = transformer.transform(transformations)
    return transformed_model


def apply_insert_before(model):
    converter = TFModelConverterFactory.create(model)

    transformations = TFTransformationLayout()
    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, signedness_to_force=None, per_channel=False)

    functional_model = is_functional_model(model)
    for i, layer in enumerate(model.layers):
        # Insertion before input layer is not supported
        if isinstance(layer, layers.InputLayer):
            continue

        original_node_name = layer.name
        if functional_model:
            _, layer_info = converter.get_layer_info_for_node(original_node_name)
            instance_idx = layer_info.instance_idx
        else:
            instance_idx = 0

        is_keras_tensor = isinstance(layer.input, tf_internals.keras_engine.keras_tensor.KerasTensor)
        inputs = [layer.input] if is_keras_tensor else layer.input

        for port, _ in enumerate(inputs):
            fake_quantize_name = f"FakeQuantize_{i}.{port}/{original_node_name}"
            fake_quantize_layer = FakeQuantize(
                TFQuantizerSpec.from_config(qconfig, narrow_range=False, half_range=False), name=fake_quantize_name
            )

            transformations.register(
                TFInsertionCommand(
                    target_point=commands.TFBeforeLayer(
                        original_node_name, instance_idx=instance_idx, input_port_id=port
                    ),
                    callable_object=fake_quantize_layer,
                    priority=TransformationPriority.QUANTIZATION_PRIORITY,
                )
            )

    transformer = TFModelTransformer(model)
    transformed_model = transformer.transform(transformations)
    return transformed_model


def check_graphs(model, ref_graph_filename):
    tf_version = version.parse(tf.__version__).base_version
    tf_version_major, tf_version_minor = tuple(map(int, tf_version.split(".")))[:2]
    data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "model_transormer", f"{tf_version_major}.{tf_version_minor}"
    )
    ref_graph_path = os.path.abspath(os.path.join(data_dir, ref_graph_filename))

    graph, graph_to_layer_var_names_map = keras_model_to_tf_graph(model)

    if tf_version_major == 2 and tf_version_minor >= 8:
        graph = remove_node_by_name("NoOp", graph)

    nx_graph = get_nx_graph_from_tf_graph(graph, graph_to_layer_var_names_map)

    if not os.path.exists(ref_graph_path) and os.getenv("NNCF_TEST_REGEN_DOT") is not None:
        write_dot_graph(nx_graph, ref_graph_path)

    compare_nx_graph_with_reference(nx_graph, ref_graph_path)


def test_functional_insert_after():
    model = create_functional_model()
    transformed_model = apply_insert_after(model)

    check_graphs(transformed_model, "functional_insert_after.dot")


def test_functional_insert_before():
    model = create_functional_model()
    transformed_model = apply_insert_before(model)

    check_graphs(transformed_model, "functional_insert_before.dot")


def test_sequential_insert_after():
    model = create_sequential_model()
    transformed_model = apply_insert_after(model)

    check_graphs(transformed_model, "sequential_block_insert_after.dot")


def test_sequential_insert_before():
    model = create_sequential_model()
    transformed_model = apply_insert_before(model)

    check_graphs(transformed_model, "sequential_block_insert_before.dot")
