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

# pylint: disable=no-member, redefined-outer-name, no-name-in-module

from dataclasses import dataclass
from typing import List
from contextlib import nullcontext as does_not_raise

import numpy as np
import onnx
import pytest

from tests.onnx.quantization.common import infer_model, min_max_quantize_model, ptq_quantize_model


def create_initializer_tensor(
        name: str,
        tensor_array: np.ndarray,
        data_type: onnx.TensorProto = onnx.TensorProto.FLOAT
) -> onnx.TensorProto:
    # (TensorProto)
    initializer_tensor = onnx.helper.make_tensor(
        name=name,
        data_type=data_type,
        dims=tensor_array.shape,
        vals=tensor_array.flatten().tolist())

    return initializer_tensor


@dataclass
class TestCase:
    input_shape: List[int]
    model: onnx.ModelProto


@pytest.fixture
def fxt_reshape_weight_graph():
    # This graph pattern is in inception-v1-12:
    # https://github.com/onnx/models/tree/main/vision/classification/inception_and_googlenet/inception_v1
    #
    #       X + Z      reshaped_W
    #          \     /
    #           GEMM
    #             |
    #          softmax

    # IO tensors (ValueInfoProto).
    model_input_name = "X"
    model_input_channels = 10
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, model_input_channels])
    model_output_name = "Y"
    model_output_channels = 5
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           [1, model_output_channels])

    w_tensor = create_initializer_tensor(
        name="W",
        tensor_array=np.random.standard_normal(
            [1, 1, model_input_channels, model_output_channels]),
        data_type=onnx.TensorProto.FLOAT)

    w_shape_tensor = create_initializer_tensor(
        name="w_shape",
        tensor_array=np.array([model_input_channels, model_output_channels]),
        data_type=onnx.TensorProto.INT64)

    z_tensor = create_initializer_tensor(
        name="z_tensor",
        tensor_array=np.random.standard_normal([1, model_input_channels]),
        data_type=onnx.TensorProto.FLOAT)

    reshaped_w_node = onnx.helper.make_node(
        op_type="Reshape",
        inputs=["W", "w_shape"],
        outputs=["reshaped_w"],
    )

    added_x_node = onnx.helper.make_node(
        op_type="Add",
        inputs=["X", "z_tensor"],
        outputs=["added_x"],
    )

    gemm_node = onnx.helper.make_node(
        'Gemm',
        inputs=['added_x', 'reshaped_w'],
        outputs=['logit']
    )

    softmax_node = onnx.helper.make_node(
        'Softmax',
        inputs=['logit'],
        outputs=['Y'],
    )

    graph_def = onnx.helper.make_graph(
        nodes=[reshaped_w_node, added_x_node, gemm_node, softmax_node],
        name="Net",
        inputs=[X],
        outputs=[Y],
        initializer=[w_tensor, w_shape_tensor, z_tensor],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    yield TestCase(input_shape=[1, model_input_channels], model=model_def)


@pytest.fixture
def fxt_weight_sharing_graph():
    # This graph pattern is in retinanet-9:
    # https://github.com/onnx/models/tree/main/vision/object_detection_segmentation/retinanet
    #
    #             X
    #             |
    #           ReLU
    #          /    \
    # W -> Conv1    Conv2 <- W
    #          \    /
    #           Add
    #            |
    #            Y
    input_shape = output_shape = [1, 1, 5, 5]

    # IO tensors (ValueInfoProto).
    model_input_name = "X"
    X = onnx.helper.make_tensor_value_info(model_input_name,
                                           onnx.TensorProto.FLOAT,
                                           input_shape)
    model_output_name = "Y"
    Y = onnx.helper.make_tensor_value_info(model_output_name,
                                           onnx.TensorProto.FLOAT,
                                           output_shape)

    W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                    [1., 1., 1.],
                    [1., 1., 1.]]]]).astype(np.float32)

    w_tensor = create_initializer_tensor(
        name="W",
        tensor_array=W,
        data_type=onnx.TensorProto.FLOAT)

    relu_x_node = onnx.helper.make_node(
        "Relu",
        inputs=["X"],
        outputs=["relu_X"],
    )

    conv1_node = onnx.helper.make_node(
        name="Conv1",
        op_type="Conv",
        inputs=["relu_X", "W"],
        outputs=["conv_1"],
        kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[1, 1, 1, 1],
    )

    conv2_node = onnx.helper.make_node(
        name="Conv2",
        op_type="Conv",
        inputs=["relu_X", "W"],
        outputs=["conv_2"],
        kernel_shape=[3, 3],
        # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
        pads=[1, 1, 1, 1],
    )

    add_node = onnx.helper.make_node(
        op_type="Add",
        inputs=["conv_1", "conv_2"],
        outputs=["Y"],
    )

    graph_def = onnx.helper.make_graph(
        nodes=[relu_x_node, conv1_node, conv2_node, add_node],
        name="Net",
        inputs=[X],
        outputs=[Y],
        initializer=[w_tensor],
    )

    # Create the model (ModelProto)
    model_def = onnx.helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 13

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    yield TestCase(input_shape=input_shape, model=model_def)


@pytest.fixture
def fxt_old_opset_version(fxt_weight_sharing_graph):
    fxt_weight_sharing_graph.model.opset_import[0].version = 12
    yield fxt_weight_sharing_graph


def min_max_quantize_model_for_fxt_graph(fxt_graph: TestCase, convert_opset_version: bool = True):
    input_shape = fxt_graph.input_shape
    model = fxt_graph.model

    with does_not_raise():
        quantized_model = min_max_quantize_model(
            input_shape, model, convert_opset_version)
        infer_model(input_shape, quantized_model)


def ptq_quantize_model_for_fxt_graph(fxt_graph: TestCase, convert_opset_version: bool = True):
    input_shape = fxt_graph.input_shape
    model = fxt_graph.model

    with does_not_raise():
        quantized_model = ptq_quantize_model(
            input_shape, model, convert_opset_version)
        infer_model(input_shape, quantized_model)


def test_fxt_reshape_weight_graph(fxt_reshape_weight_graph: TestCase):
    min_max_quantize_model_for_fxt_graph(fxt_reshape_weight_graph)
    ptq_quantize_model_for_fxt_graph(fxt_reshape_weight_graph)


def test_fxt_weight_sharing_graph(fxt_weight_sharing_graph: TestCase):
    min_max_quantize_model_for_fxt_graph(fxt_weight_sharing_graph)
    ptq_quantize_model_for_fxt_graph(fxt_weight_sharing_graph)


@pytest.mark.parametrize("convert_opset_version", [True, False])
def test_fxt_old_opset_version(fxt_old_opset_version: TestCase, convert_opset_version: bool):
    min_max_quantize_model_for_fxt_graph(fxt_old_opset_version, convert_opset_version)
    ptq_quantize_model_for_fxt_graph(fxt_old_opset_version, convert_opset_version)
