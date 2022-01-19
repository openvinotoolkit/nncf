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

from typing import List
from typing import TypeVar

import onnx
import numpy as np

NodeType = TypeVar('NodeType')
InitializerType = TypeVar('InitializerType')


# pylint: disable=no-member

class ONNXGraph:
    """
    The class stores onnx model and provides the interface to interact with it.
    """

    def __init__(self, onnx_model: onnx.ModelProto):
        self.onnx_model = onnx_model
        self.model_with_shapes = onnx.shape_inference.infer_shapes(self.onnx_model)

    def get_nodes_by_output(self, output_name: str) -> List[NodeType]:
        """
        Returns all nodes that have output with the name 'output_name'.
        """
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if output_name in node.output or output_name == node.output:
                output.append(node)
        return output

    def get_nodes_by_input(self, input_name: str) -> List[NodeType]:
        """
        Returns all nodes that have input with the name 'output_name'.
        """
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if input_name in node.input or input_name == node.input:
                output.append(node)
        return output

    def get_all_nodes(self) -> List[NodeType]:
        """
        Returns all nodes of onnx model.
        """
        return self.onnx_model.graph.node

    def get_all_model_inputs(self) -> List[InitializerType]:
        """
        Returns all model inputs.
        """
        return self.onnx_model.graph.input

    def get_all_model_outputs(self) -> List[InitializerType]:
        """
        Returns all model outputs.
        """
        return self.onnx_model.graph.output

    def get_all_node_inputs(self, node_name: str) -> List[InitializerType]:
        """
        Returns all node input edges.
        """
        node_inputs = None
        graph = self.onnx_model.graph
        for node in graph.node:
            if node.name == node_name:
                node_inputs = node.input
        return node_inputs

    def get_all_node_outputs(self, node_name: str) -> List[InitializerType]:
        """
        Returns all node input edges.
        """
        node_outputs = None
        graph = self.onnx_model.graph
        for node in graph.node:
            if node.name == node_name:
                node_outputs = node.output
        return node_outputs

    def get_nodes_by_type(self, node_type: str) -> List[NodeType]:
        """
        Returns all nodes in the model that have type equal to 'node_type'.
        """
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if str(node.op_type) == node_type:
                output.append(node)
        return output

    def find_weight_input_in_module(self, node_name: str) -> InitializerType:
        """
        Returns weight Initializaer of the mode with the name 'node_name'.
        """
        node_inputs = self.get_all_node_inputs(node_name)
        # TODO(kshpv): add search of input weight tensor
        return node_inputs[1]

    def get_initializers_value(self, initializer_name: str) -> np.ndarray:
        """
        Returns tensor value of model's Initializer with the name equals to 'initializer_name'.
        """
        tensor = None
        graph = self.onnx_model.graph
        for init in graph.initializer:
            if init.name == initializer_name:
                tensor = onnx.numpy_helper.to_array(init)
        return tensor

    def get_node_output_shape(self, node_output_name: str) -> List[int]:
        """
        Returns node's output shape with output name equals to node_output_name.
        """
        shape = []
        activations_shapes = self.model_with_shapes.graph.value_info
        for tensor in activations_shapes:
            if tensor.name == node_output_name:
                for dim in tensor.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value)
        return shape
