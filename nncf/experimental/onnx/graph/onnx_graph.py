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
import itertools

import onnx
from onnx import NodeProto  # pylint: disable=no-name-in-module
from onnx import ValueInfoProto  # pylint: disable=no-name-in-module
from onnx import numpy_helper  # pylint: disable=no-name-in-module
import numpy as np


# pylint: disable=no-member, too-many-public-methods

class ONNXGraph:
    """
    The class stores onnx model and provides the interface to interact with it.
    """

    @staticmethod
    def get_all_nodes(model: onnx.ModelProto) -> List[NodeProto]:
        """
        Returns all nodes of onnx model.
        """
        return model.graph.node

    @staticmethod
    def get_node_by_name(model: onnx.ModelProto, node_name: str) -> Optional[NodeProto]:
        for node in model.graph.node:
            if node.name == node_name:
                return node
        return None

    @staticmethod
    def get_model_inputs(model: onnx.ModelProto) -> List[ValueInfoProto]:
        """
        Returns model inputs.
        """
        inputs = []
        input_all = [node.name for node in model.graph.input]
        input_initializer = [node.name for node in model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        for node in model.graph.input:
            if node.name in net_feed_input:
                inputs.append(node)
        return inputs

    @staticmethod
    def get_model_outputs(model: onnx.ModelProto) -> List[ValueInfoProto]:
        """
        Returns model outputs.
        """
        return list(model.graph.output)

    @staticmethod
    def get_nodes_by_output(model: onnx.ModelProto, output_name: str) -> List[NodeProto]:
        """
        Returns all nodes that have output with the name 'output_name'.
        """
        return ONNXGraph._get_nodes_by_lambda(model, output_name, lambda node: node.output)

    @staticmethod
    def get_nodes_by_input(model: onnx.ModelProto, input_name: str) -> List[NodeProto]:
        """
        Returns all nodes that have input with the name 'input_name'.
        """
        return ONNXGraph._get_nodes_by_lambda(model, input_name, lambda node: node.input)

    @staticmethod
    def _get_nodes_by_lambda(model: onnx.ModelProto, name: str, func: Callable[[NodeProto], List[NodeProto]]):
        output = []
        for node in model.graph.node:
            if name in func(node) or name == func(node):
                output.append(node)
        return output

    @staticmethod
    def get_node_edges(model: onnx.ModelProto, node_name: str) -> Dict[str, List[ValueInfoProto]]:
        """
        Returns node input and output edges.
        """
        for node in model.graph.node:
            if node.name == node_name:
                return {'input': list(node.input),
                        'output': list(node.output)}
        raise RuntimeError('There is no node with the name {}'.format(node_name))

    @staticmethod
    def get_input_port_id_for_node_after_input(input_name: str, to_node: NodeProto) -> int:
        """
        Returns input_port_id for 'to_node' connected with the model input with the name 'input_name'.
        :param input_name: Name of the ONNX model Input.
        :param to_node: Node, which has input edge with 'input_name' name.
        :return: input port number for 'to_node', which is connected to 'input_name'.
        """
        for input_port_id, port in enumerate(to_node.input):
            if port == input_name:
                return input_port_id
        raise RuntimeError(f'The node {to_node} does not have input edge with the name {input_name}')

    @staticmethod
    def get_output_port_id_for_node_before_output(output_name: str, from_node: NodeProto) -> int:
        """
        Returns output_port_id for 'from_node' connected with the model output with the name 'output_name'.
        :param output_name: Name of the ONNX model Output.
        :param from_node: Node, which has output edge with 'output_name' name.
        :return: output port number for 'from_node', which is connected to 'output_name'.
        """
        for output_port_id, port in enumerate(from_node.output):
            if port == output_name:
                return output_port_id
        raise RuntimeError(f'The node {from_node} does not have output edge with the name {output_name}')

    @staticmethod
    def get_port_ids_between_nodes(from_node: NodeProto, to_node: NodeProto) -> Dict[str, int]:
        """
        Returns input_port_id and output_port_id between 'from_node' and 'to_node'.
        :param from_node: Node, whose output is connected to 'to_node' node.
        :param to_node: Node, whose input is connected to 'from_node' node.
        :return: Dict{'input_port_id': input port id, 'output_port_id': output port id}
        """
        output = {'input_port_id': None, 'output_port_id': None}
        for port_id, port in enumerate(to_node.input):
            if port in from_node.output:
                output['input_port_id'] = port_id
        for port_id, port in enumerate(from_node.output):
            if port in to_node.input:
                output['output_port_id'] = port_id
        if output['output_port_id'] is None or output['input_port_id'] is None:
            raise RuntimeError(f'The nodes {from_node.name} and {to_node.name} do not have edges between.')
        return output

    @staticmethod
    def get_nodes_by_type(model: onnx.ModelProto, node_type: str) -> List[NodeProto]:
        """
        Returns all nodes in the model that have type equal to 'node_type'.
        """
        output = []
        for node in model.graph.node:
            if str(node.op_type) == node_type:
                output.append(node)
        return output

    @staticmethod
    def get_weight_tensor_with_initializer(model: onnx.ModelProto, node_name: str) -> Optional[str]:
        """
        Return 'node_name' node's input weight tensor if it has an initializer type.
        Otherwise, return None.
        """
        node_inputs = ONNXGraph.get_node_edges(model, node_name)['input']

        # TODO(kshpv): add search of input weight tensor
        weight_tensor_name = node_inputs[1]
        return weight_tensor_name

    @staticmethod
    def get_weight_input_in_module(model: onnx.ModelProto, node_name: str) -> ValueInfoProto:
        """
        Returns 'node_name' node's input weight tensor.
        """
        node_inputs = ONNXGraph.get_node_edges(model, node_name)['input']
        # TODO(kshpv): add search of input weight tensor
        return node_inputs[1]

    @staticmethod
    def get_node_index(model: onnx.ModelProto, node_name: str):
        for i, node in enumerate(ONNXGraph.get_all_nodes(model)):
            if node.name == node_name:
                return i
        return -1

    @staticmethod
    def get_initializers_value(model: onnx.ModelProto, initializer_name: str) -> np.ndarray:
        """
        Returns tensor value of model's Initializer with the name equals to 'initializer_name'.
        """
        for init in model.graph.initializer:
            if init.name == initializer_name:
                tensor = numpy_helper.to_array(init)
                return tensor
        raise RuntimeError('There is no initializer with the name {}'.format(initializer_name))

    @staticmethod
    def get_initializer(model: onnx.ModelProto, initializer_name: str) -> np.ndarray:
        """
        Returns model's Initializer with the name equals to 'initializer_name'.
        """
        for init in model.graph.initializer:
            if init.name == initializer_name:
                return init
        raise RuntimeError('There is no initializer with the name {}'.format(initializer_name))

    @staticmethod
    def get_tensor_shape(tensor: ValueInfoProto) -> List[int]:
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

    @staticmethod
    def get_edge_shape(model: onnx.ModelProto, edge_name: str) -> List[int]:
        """
        Returns tensor shape of the edge with the name 'edge_name'.
        """
        activations_tensors = model.graph.value_info
        inputs = model.graph.input
        outputs = model.graph.output
        for tensor in itertools.chain(activations_tensors, inputs, outputs):
            if tensor.name == edge_name:
                return ONNXGraph.get_tensor_shape(tensor)
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))

    @staticmethod
    def get_edge_dtype(model: onnx.ModelProto, edge_name: str) -> int:
        """
        Returns the data type of the edge with the name 'edge_name'.
        """
        activations_tensors = model.graph.value_info
        inputs = model.graph.input
        outputs = model.graph.output
        for tensor in itertools.chain(activations_tensors, inputs, outputs):
            if tensor.name == edge_name:
                return tensor.type.tensor_type.elem_type
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))

    @staticmethod
    def get_edge_dtype_name(model: onnx.ModelProto, edge_name: str) -> str:
        """
        Returns the name of datatype of the edge with the name 'edge_name'.
        """
        return onnx.TensorProto.DataType.Name(ONNXGraph.get_edge_dtype(model, edge_name))
