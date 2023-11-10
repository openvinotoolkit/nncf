# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod
from typing import List

import numpy as np
import onnx
import pytest

from nncf.common.graph.layer_attributes import MultipleInputLayerAttributes
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.nncf_graph_builder import ONNXLayerAttributes
from tests.onnx.models import OPSET_VERSION
from tests.onnx.models import create_initializer_tensor


class ONNXNodeCreator:
    def __init__(self, op_name):
        self._initializers = []
        self._node = None
        self._op_name = op_name

    @property
    def initializers(self):
        return self._initializers

    @property
    def node(self):
        return self._node

    @property
    def input_shape(self):
        return [3, 3, 3]

    @property
    def output_shape(self):
        return self.input_shape

    @property
    def num_inputs(self):
        return 1

    @property
    def num_outputs(self):
        return 1

    @abstractmethod
    def build_one_node(self, model_input_names: List[str], model_outptu_names: List[str]):
        pass

    def get_one_layer_model(self):
        model_input_names = [f"X_{i}" for i in range(self.num_inputs)]
        model_output_names = [f"Y_{i}" for i in range(self.num_outputs)]

        inputs = [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, self.input_shape)
            for name in model_input_names
        ]
        outputs = [
            onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, self.output_shape)
            for name in model_output_names
        ]

        self.build_one_node(model_input_names, model_output_names)

        graph_def = onnx.helper.make_graph(
            nodes=[self.node],
            name="OneLayerNet",
            inputs=inputs,
            outputs=outputs,
            initializer=self.initializers,
        )

        op = onnx.OperatorSetIdProto()
        op.version = OPSET_VERSION
        model = onnx.helper.make_model(graph_def, opset_imports=[op])
        onnx.checker.check_model(model)
        return model


class ONNXConvCreator(ONNXNodeCreator):
    def build_one_node(self, input_names, output_names):
        input_name, output_name = input_names[0], output_names[0]
        in_ch = self.input_shape[0]
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
            name=self._op_name,
            op_type="Conv",
            inputs=[input_name, conv1_W_initializer_tensor_name, conv1_B_initializer_tensor_name],
            outputs=[output_name],
            kernel_shape=conv1_kernel_shape,
        )


class ONNXIdentityCreator(ONNXNodeCreator):
    def build_one_node(self, input_names, output_names):
        self._node = onnx.helper.make_node(
            name=self._op_name, op_type="Identity", inputs=input_names, outputs=output_names
        )


class ONNXConcatCreator(ONNXNodeCreator):
    def build_one_node(self, input_names, output_names):
        self._node = onnx.helper.make_node(
            name=self._op_name,
            op_type="Concat",
            inputs=input_names,
            outputs=output_names,
            axis=2,
        )

    @property
    def input_shape(self):
        return [1, 2, 2, 2]

    @property
    def output_shape(self):
        return [1, 2, 6, 2]

    @property
    def num_inputs(self):
        return 3


@pytest.mark.parametrize(
    "node_creator, ref_layer_attrs",
    [
        (ONNXIdentityCreator, ONNXLayerAttributes()),
        (
            ONNXConvCreator,
            ONNXLayerAttributes(
                weight_attrs={1: {"name": "Conv1_W", "shape": [3, 3, 1, 1]}}, bias_attrs={"name": "Conv1_B"}
            ),
        ),
        (
            ONNXConcatCreator,
            ONNXLayerAttributes(
                weight_attrs={},
                bias_attrs={},
                layer_attributes=MultipleInputLayerAttributes(axis=2, num_inputs=3),
            ),
        ),
    ],
)
def test_layer_attributes(node_creator: ONNXNodeCreator, ref_layer_attrs):
    op_name = "test_node"
    onnx_model = node_creator(op_name).get_one_layer_model()
    nncf_graph = GraphConverter.create_nncf_graph(onnx_model)
    node = nncf_graph.get_node_by_name(op_name)
    assert node.layer_attributes.__dict__ == ref_layer_attrs.__dict__
