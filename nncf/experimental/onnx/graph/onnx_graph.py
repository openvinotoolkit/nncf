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

from typing import Callable
from typing import Dict
from typing import List

import onnx
from onnx import NodeProto  # pylint: disable=no-name-in-module
from onnx import ValueInfoProto  # pylint: disable=no-name-in-module
from onnx import numpy_helper
import numpy as np


# pylint: disable=no-member

class ONNXGraph:
    """
    The class stores onnx model and provides the interface to interact with it.
    """

    def __init__(self, onnx_model: onnx.ModelProto):
        self.onnx_model = onnx_model
        self.model_with_shapes = onnx.shape_inference.infer_shapes(self.onnx_model)
        self.activations_tensors = self.model_with_shapes.graph.value_info
        inputs = self.model_with_shapes.graph.input
        outputs = self.model_with_shapes.graph.output
        self.activations_tensors.extend(inputs)
        self.activations_tensors.extend(outputs)

    def get_all_nodes(self) -> List[NodeProto]:
        """
        Returns all nodes of onnx model.
        """
        return self.onnx_model.graph.node

    def get_model_inputs(self) -> List[ValueInfoProto]:
        """
        Returns model inputs.
        """
        inputs = []
        input_all = [node.name for node in self.onnx_model.graph.input]
        input_initializer = [node.name for node in self.onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        for node in self.onnx_model.graph.input:
            if node.name in net_feed_input:
                inputs.append(node)
        return inputs

    def get_model_outputs(self) -> List[ValueInfoProto]:
        """
        Returns model outputs.
        """
        return list(self.onnx_model.graph.output)

    def get_nodes_by_output(self, output_name: str) -> List[NodeProto]:
        """
        Returns all nodes that have output with the name 'output_name'.
        """
        return self._get_nodes_by_lambda(output_name, lambda node: node.output)

    def get_nodes_by_input(self, input_name: str) -> List[NodeProto]:
        """
        Returns all nodes that have input with the name 'input_name'.
        """
        nodes = self._get_nodes_by_lambda(input_name, lambda node: node.input)
        if len(nodes) == 0:
            raise RuntimeError(f'There is no nodes with the input {input_name}')
        return nodes

    def _get_nodes_by_lambda(self, name: str, func: Callable[[NodeProto], List[NodeProto]]):
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if name in func(node) or name == func(node):
                output.append(node)
        return output

    def get_node_edges(self, node_name: str) -> Dict[str, List[ValueInfoProto]]:
        """
        Returns node input and output edges.
        """
        graph = self.onnx_model.graph
        for node in graph.node:
            if node.name == node_name:
                return {'input': list(node.input),
                        'output': list(node.output)}
        raise RuntimeError('There is no node with the name {}'.format(node_name))

    def get_nodes_by_type(self, node_type: str) -> List[NodeProto]:
        """
        Returns all nodes in the model that have type equal to 'node_type'.
        """
        output = []
        graph = self.onnx_model.graph
        for node in graph.node:
            if str(node.op_type) == node_type:
                output.append(node)
        return output

    def find_weight_input_in_module(self, node_name: str) -> ValueInfoProto:
        """
        Returns weight Initializaer of the mode with the name 'node_name'.
        """
        node_inputs = self.get_node_edges(node_name)['input']
        # TODO(kshpv): add search of input weight tensor
        return node_inputs[1]

    def get_initializers_value(self, initializer_name: str) -> np.ndarray:
        """
        Returns tensor value of model's Initializer with the name equals to 'initializer_name'.
        """
        graph = self.onnx_model.graph
        for init in graph.initializer:
            if init.name == initializer_name:
                tensor = numpy_helper.to_array(init)
                return tensor
        raise RuntimeError('There is no initializer with the name {}'.format(initializer_name))

    def get_tensor_shape(self, tensor: ValueInfoProto) -> List[int]:
        """
        Returns 'tensor' shape.
        """
        tensor_type = tensor.type.tensor_type
        shape = []
        if tensor_type.HasField("shape"):
            for d in tensor_type.shape.dim:
                if d.HasField("dim_value"):
                    dim_value = d.dim_value
                    if isinstance(dim_value, int):
                        shape.append(dim_value)
                    else:
                        raise RuntimeError(f'The tensor {tensor.name} has non integer shape.')
                else:
                    raise RuntimeError(f'The tensor {tensor.name} does not have dim_value field.')
        else:
            raise RuntimeError(f'The tensor {tensor.name} does not have shape field')
        return shape

    def get_edge_shape(self, edge_name: str) -> List[int]:
        """
        Returns tensor shape of the edge with the name 'edge_name'.
        """
        for tensor in self.activations_tensors:
            if tensor.name == edge_name:
                return self.get_tensor_shape(tensor)
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))

    def get_edge_dtype(self, edge_name: str) -> str:
        """
        Returns the data type of the edge with the name 'edge_name'.
        """
        for tensor in self.activations_tensors:
            if tensor.name == edge_name:
                elem_type = tensor.type.tensor_type.elem_type
                return onnx.TensorProto.DataType.Name(elem_type)
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))
