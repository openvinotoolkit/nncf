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
import pytest

from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from tests.onnx.models import OPSET_VERSION
from tests.onnx.models import create_initializer_tensor


class ONNXNodeCreator:
    def __init__(self):
        self._initializers = []
        self._node = None

    @property
    def initializers(self):
        return self._initializers

    @property
    def node(self):
        return self._node


class ONNXConvCreator(ONNXNodeCreator):
    def __init__(self, node_name, input_name, output_name, input_shape):
        super().__init__()
        in_ch = input_shape[0]
        conv1_in_channels, conv1_out_channels, conv1_kernel_shape = in_ch, in_ch, (1, 1)

        conv1_W = np.ones(shape=(conv1_out_channels, conv1_in_channels, *conv1_kernel_shape))
        conv1_B = np.ones(shape=conv1_out_channels)

        conv1_W_initializer_tensor_name = "Conv1_W"
        conv1_W_initializer_tensor = create_initializer_tensor(
            name=conv1_W_initializer_tensor_name, tensor_array=conv1_W, data_type=onnx.TensorProto.FLOAT
        )
        conv1_B_initializer_tensor_name = "Conv1_B"
        conv1_B_initializer_tensor = create_initializer_tensor(
            name=conv1_B_initializer_tensor_name, tensor_array=conv1_B, data_type=onnx.TensorProto.FLOAT
        )
        self._initializers = [conv1_W_initializer_tensor, conv1_B_initializer_tensor]

        self._node = onnx.helper.make_node(
            name=node_name,
            op_type="Conv",
            inputs=[input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[output_name],
            kernel_shape=conv1_kernel_shape,
        )


class ONNXIdentityCreator(ONNXNodeCreator):
    def __init__(self, node_name, input_name, output_name, input_shape):
        super().__init__()
        self._node = onnx.helper.make_node(
            name=node_name, op_type="Identity", inputs=[input_name], outputs=[output_name]
        )


def get_one_layer_model(op_name: str, node_creator: ONNXNodeCreator, input_shape):
    model_input_name = "X"
    model_output_name = "Y"
    X = onnx.helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, input_shape)

    Y = onnx.helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, input_shape)

    node_desc = node_creator(op_name, model_input_name, model_output_name, input_shape)
    graph_def = onnx.helper.make_graph(
        nodes=[node_desc.node],
        name="ConvNet",
        inputs=[X],
        outputs=[Y],
        initializer=node_desc.initializers,
    )

    op = onnx.OperatorSetIdProto()
    op.version = OPSET_VERSION
    model = onnx.helper.make_model(graph_def, opset_imports=[op])
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize(
    "node_creator, ref_layer_attrs",
    [
        (ONNXIdentityCreator, ONNXLayerAttributes()),
        (
            ONNXConvCreator,
            ONNXLayerAttributes(
                weight_attrs={1: {"name": "Conv1_W", "shape": [3, 3, 1, 1]}},
                bias_attrs={"name": "Conv1_B"},
            ),
        ),
    ],
)
def test_layer_attributes(node_creator, ref_layer_attrs):
    input_shape = [3, 3, 3]
    op_name = "test_node"
    onnx_model = get_one_layer_model(op_name, node_creator, input_shape)
    nncf_graph = GraphConverter.create_nncf_graph(onnx_model)
    node = nncf_graph.get_node_by_name(op_name)
    assert node.layer_attributes.__dict__ == ref_layer_attrs.__dict__
