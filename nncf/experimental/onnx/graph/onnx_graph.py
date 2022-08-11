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

from typing import Callable, Optional
from typing import Dict
from typing import List

import onnx
from onnx import NodeProto  # pylint: disable=no-name-in-module
from onnx import ValueInfoProto  # pylint: disable=no-name-in-module
from onnx import numpy_helper  # pylint: disable=no-name-in-module
import numpy as np

from skl2onnx.helpers.onnx_helper import enumerate_model_node_outputs


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
        self.initializer_names = {n.name for n in self.onnx_model.graph.initializer}
        self.lookup_nodes = {n.name: n for n in self.onnx_model.graph.node}
        self.valid_tensor_names = set(enumerate_model_node_outputs(onnx_model))
        self.valid_tensor_names.update({inp.name for inp in self.get_model_inputs()})

    def is_valid_tensor(self, tensor_name: str) -> bool:
        """
        If tensor_name is not the output of all nodes or the model input, return True.
        Otherwise, return False.
        """
        if tensor_name in self.valid_tensor_names:
            return True

        return False

    def get_all_nodes(self) -> List[NodeProto]:
        """
        Returns all nodes of onnx model.
        """
        return self.onnx_model.graph.node

    def get_node_by_name(self, node_name: str) -> NodeProto:
        try:
            return self.lookup_nodes[node_name]
        except KeyError as e:
            raise KeyError(f"There is no node with the name {node_name}") from e

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

    def get_weight_tensor_with_initializer(self, node_name: str) -> Optional[str]:
        """
        Return 'node_name' node's input weight tensor if it has an initializer type.
        Otherwise, return None.
        """
        node_inputs = self.get_node_edges(node_name)['input']

        # TODO(kshpv): add search of input weight tensor
        weight_tensor_name = node_inputs[1]

        if weight_tensor_name in self.initializer_names:
            return weight_tensor_name
        raise RuntimeError(
            'There is no weight tensor with the name {}.'
            ' Probably this node utilizes other nodes outputs as its weight '.format(
                node_name))

    def get_weight_input_in_module(self, node_name: str) -> ValueInfoProto:
        """
        Returns 'node_name' node's input weight tensor.
        """
        node_inputs = self.get_node_edges(node_name)['input']
        # TODO(kshpv): add search of input weight tensor
        return node_inputs[1]

    def get_node_index(self, node_name: str):
        for i, node in enumerate(self.get_all_nodes()):
            if node.name == node_name:
                return i
        return -1

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
                elif d.HasField("dim_param"):
                    # flexible shape
                    # make manually 1
                    shape.append(1)
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

    def get_edge_dtype(self, edge_name: str) -> int:
        """
        Returns the data type of the edge with the name 'edge_name'.
        """
        for tensor in self.activations_tensors:
            if tensor.name == edge_name:
                return tensor.type.tensor_type.elem_type
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))

    def get_edge_dtype_name(self, edge_name: str) -> str:
        """
        Returns the name of datatype of the edge with the name 'edge_name'.
        """
        return onnx.TensorProto.DataType.Name(self.get_edge_dtype(edge_name))
