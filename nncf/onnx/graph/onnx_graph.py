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

from typing import Callable, Dict, List, Optional, Tuple

from collections import deque
import onnx
from onnx import numpy_helper  # pylint: disable=no-name-in-module
import numpy as np

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import OpWeightDef
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXIdentityMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXReshapeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXQuantizeLinearMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES


# pylint: disable=no-member
# pylint: disable=too-many-public-methods

class ONNXGraph:
    """
    The class provides the interface to get the necessary information from ONNX model.
    """

    def __init__(self, onnx_model: onnx.ModelProto):
        self.onnx_model = onnx_model
        self._node_name_to_node = None  # type: Dict[str, onnx.onnx.NodeProto]
        self._activations_tensor_name_to_value_info = None  # type: Dict[str, onnx.onnx.ValueInfoProto]

    def _update_activation_tensors(self, do_shape_inference: bool = False) -> None:
        if do_shape_inference:
            self.onnx_model = onnx.shape_inference.infer_shapes(self.onnx_model)
        self._activations_tensor_name_to_value_info = {tensor.name: tensor for tensor in
                                                       self.onnx_model.graph.value_info}
        model_inputs_name_to_value_info = {tensor.name: tensor for tensor in self.onnx_model.graph.input}
        model_outputs_name_to_value_info = {tensor.name: tensor for tensor in self.onnx_model.graph.output}
        self._activations_tensor_name_to_value_info.update(model_inputs_name_to_value_info)
        self._activations_tensor_name_to_value_info.update(model_outputs_name_to_value_info)

    def _update_node_names(self) -> None:
        self._node_name_to_node = {n.name: n for n in self.onnx_model.graph.node}

    def get_all_nodes(self) -> List[onnx.NodeProto]:
        """
        Returns model nodes in the original order.

        :return: model nodes.
        """
        return self.onnx_model.graph.node

    def get_node_by_name(self, node_name: str) -> Optional[onnx.NodeProto]:
        """
        Returns a model node with the name equals to 'node_name' from self._node_name_to_node.
        If the self._node_name_to_node is None, fills it with the nodes from the self.onnx_model.
        If there is no node with such name returns None.

        :param node_name: Name of the node.
        :return: None if the node with the specified name exists - otherwise returns the node.
        """
        if self._node_name_to_node is None:
            self._update_node_names()
        return self._node_name_to_node[node_name] if node_name in self._node_name_to_node else None

    def get_model_inputs(self) -> List[onnx.ValueInfoProto]:
        """
        Returns all model inputs.

        :return: Model Inputs.
        """
        inputs = []
        input_all = [node.name for node in self.onnx_model.graph.input]
        input_initializer = [node.name for node in self.onnx_model.graph.initializer]
        net_feed_input = list(set(input_all) - set(input_initializer))
        for node in self.onnx_model.graph.input:
            if node.name in net_feed_input:
                inputs.append(node)
        return inputs

    def get_model_outputs(self) -> List[onnx.ValueInfoProto]:
        """
        Returns all model outputs.

        :return: Model Outputs.
        """
        return list(self.onnx_model.graph.output)

    def get_nodes_by_output(self, output_name: str) -> List[onnx.NodeProto]:
        """
        Returns all nodes that have output edge with the name 'output_name'.

        :param output_name: The name of output edge.
        :return: Nodes with corresponding output.
        """
        return self._get_nodes_by_lambda(output_name, lambda node: node.output)

    def get_nodes_by_input(self, input_name: str) -> List[onnx.NodeProto]:
        """
        Returns all nodes that have input with the name 'input_name'.

        :param input_name: The name of input edge.
        :return: Nodes with corresponding input.
        """
        return self._get_nodes_by_lambda(input_name, lambda node: node.input)

    def _get_nodes_by_lambda(self, name: str, func: Callable[[onnx.NodeProto], List[onnx.NodeProto]]) -> List[
        onnx.NodeProto]:
        output = []
        for node in self.get_all_nodes():
            if name in func(node):
                output.append(node)
        return output

    def get_node_edge_names(self, node_name: str) -> Dict[str, List[str]]:
        """
        Returns node edge names.

        :param node_name: The name of the node.
        :return: Dict with two keys: 'input' and 'output',
        which are corresponding to input and output edges accordingly.
        """
        if self._node_name_to_node is None:
            self._update_node_names()
        if node_name in self._node_name_to_node:
            return {'input': list(self._node_name_to_node[node_name].input),
                    'output': list(self._node_name_to_node[node_name].output)}
        raise RuntimeError('There is no node with the name {}'.format(node_name))

    @staticmethod
    def get_input_port_id_for_node_after_input(input_name: str, to_node: onnx.NodeProto) -> int:
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
    def get_output_port_id_for_node_before_output(output_name: str, from_node: onnx.NodeProto) -> int:
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
    def get_port_ids_between_nodes(from_node: onnx.NodeProto, to_node: onnx.NodeProto) -> Dict[str, int]:
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

    def get_nodes_by_type(self, node_type: str) -> List[onnx.NodeProto]:
        """
        Returns all nodes in the model that have type equal to 'node_type'.

        :param node_type: Type of the nodes.
        :return: All nodes with the corresponding type.
        """
        output = []
        for node in self.get_all_nodes():
            if str(node.op_type) == node_type:
                output.append(node)
        return output

    def get_weight_tensor(self, node: onnx.NodeProto) -> Tuple[str, np.ndarray]:
        # TODO(kshpv): ticket: 101211
        """
        Returns node's weight tensor name and its value.

        :param node: Node, in which the weight tensor finally applied.
        :return: Weight tensor name and its value.
        """
        weight_port_id = self.get_weight_port_id(node)
        weight_input = self.get_node_edge_names(node.name)['input'][weight_port_id]
        if self.has_initializer(weight_input):
            return self.get_initializer(weight_input).name, self.get_initializers_value(weight_input)
        parent_node_on_weight_port = self.get_nodes_by_output(weight_input)[0]
        nodes = deque([parent_node_on_weight_port])
        while nodes:
            current_node = nodes.popleft()
            node_parents = self.get_parents(current_node)
            nodes.extendleft(node_parents)
            metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(current_node.op_type)
            if metatype in [ONNXIdentityMetatype, ONNXQuantizeLinearMetatype]:
                if self.has_initializer(current_node.input[0]):
                    return self._get_tensor_from_zero_input(current_node)
                continue
            if metatype == ONNXReshapeMetatype:
                return self._get_weight_tensor_with_reshape(current_node)
            if metatype in WEIGHT_LAYER_METATYPES:
                weight_port_id = self.get_weight_port_id(current_node)
                weight_input = self.get_node_edge_names(current_node.name)['input'][weight_port_id]
                return self.get_initializer(weight_input).name, self.get_initializers_value(weight_input)
        raise RuntimeError('Could not find the weight tensor of the node')

    @staticmethod
    def _get_weight_definitions(node: onnx.NodeProto) -> OpWeightDef:
        """
        Returns the weight_definitions of the node's metatype.

        :param node: Node from which weight definition is obtained.
        :return: weight definition of the node.
        """
        metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
        if metatype in WEIGHT_LAYER_METATYPES:
            return metatype.weight_definitions
        raise RuntimeError(f'The metatype {metatype} does not belong to a list of metatypes with a weight tensor.')

    def get_weight_port_id(self, node: onnx.NodeProto) -> int:
        """
        Returns input port id, where a weight tensor should output.

        :param node: Node, for which input port id is returned,
        :return: input port id, where a weight tensor should output.
        """
        weight_definitions = self._get_weight_definitions(node)
        if weight_definitions.weight_port_id is not None:
            return weight_definitions.weight_port_id
        raise RuntimeError(f'The metatype {node} does not have weight_port_id attribute')

    def get_weight_channel_axis(self, node: onnx.NodeProto) -> int:
        """
        Returns a channel axis for weight per-channel quantization.

        :param node: Node, for which weight per-channel axis id is returned,
        :return: Channel axis for per-channel quantization.
        """
        weight_definitions = self._get_weight_definitions(node)
        if weight_definitions.weight_channel_axis is not None:
            return weight_definitions.weight_channel_axis
        raise RuntimeError(f'The node {node} does not have weight_channel_axis attribute')

    def get_bias_tensor_port_id(self, node: onnx.NodeProto) -> int:
        """
        Returns input port id, where a bias tensor should output.

        :param node: Node, for which input port id is returned,
        :return: input port id, where a weight bias should output.
        """
        weight_definitions = self._get_weight_definitions(node)
        if weight_definitions.bias_port_id is not None:
            return weight_definitions.bias_port_id
        raise RuntimeError(f'The node {node} does not have bias_port_id attribute')

    def _get_weight_tensor_with_reshape(self, node: onnx.NodeProto) -> Tuple[str, np.ndarray]:
        """
        Returns node's weight tensor name and its value in the case when reshape node is placed after the weight.
        The returned weight tensor will be reshaped according to a shape attribute of the reshape node.

        :param node: Reshape node, whose input is weight tensor.
        :return: The weight tensor name and its value with applied the reshape operation.
        """
        tensor_name = node.output[0]
        shape = self.get_initializers_value(node.input[1])
        tensor_value = self.get_initializers_value(node.input[0])
        reshaped_tensor_value = tensor_value.reshape(shape)
        return tensor_name, reshaped_tensor_value

    def _get_tensor_from_zero_input(self, node: onnx.NodeProto) -> Tuple[str, np.ndarray]:
        """
        Returns the weight tensor name and its value, which is located on the 0-index input port of the node.

        :param node: Node, which takes on the 0-index input port id the weight tensor.
        :return: The weight tensor name and its value.
        """
        tensor_name = self.get_initializer(node.input[0]).name
        return tensor_name, self.get_initializers_value(tensor_name)

    def get_node_index(self, node_name: str) -> int:
        """
        Returns the node index in the model.

        :param node_name: Name of the node.
        :return: Node index, -1 if there is no such node.
        """
        for i, node in enumerate(self.get_all_nodes()):
            if node.name == node_name:
                return i
        return -1

    def get_initializers_value(self, initializer_name: str) -> np.ndarray:
        """
        Returns tensor value of model's Initializer with the name equals to 'initializer_name'.

        :param initializer_name: Name of the tensor.
        :return: The value of the tensor.
        """
        for init in self.onnx_model.graph.initializer:
            if init.name == initializer_name:
                tensor = numpy_helper.to_array(init)
                return tensor
        raise RuntimeError('There is no initializer with the name {}'.format(initializer_name))

    def has_initializer(self, initializer_name: str) -> bool:
        """
        Returns True whether the model has the initializer with the name equals to initializer_name.

        :param initializer_name: Name of the initializer.
        :return: True if the model has such initializer, False - otherwise.
        """
        for init in self.onnx_model.graph.initializer:
            if init.name == initializer_name:
                return True
        return False

    def get_initializer(self, initializer_name: str) -> onnx.TensorProto:
        """
        Returns model's Initializer with the name equals to 'initializer_name'.

        :param initializer_name: Name of the Initializer.
        :return: The Initializer.
        """
        for init in self.onnx_model.graph.initializer:
            if init.name == initializer_name:
                return init
        raise RuntimeError('There is no initializer with the name {}'.format(initializer_name))

    @staticmethod
    def get_tensor_shape(tensor: onnx.ValueInfoProto) -> List[int]:
        """
        Returns 'tensor' shape.

        :param tensor: The tensor.
        :return: Shape of the Tensor.
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
        Returns a shape of the edge with the name 'edge_name'.
        If the activations tensors were not filled in self._activations_tensor_name_to_value_info, it updates them.
        If after updating of the self._activations_tensor_name_to_value_info, there is still no such tensor,
        do shape inference of the model.

        :param edge_name: The name of the edge.
        :return: Shape of the tensor on that edge.
        """
        if self._activations_tensor_name_to_value_info is None:
            self._update_activation_tensors()
        if edge_name in self._activations_tensor_name_to_value_info:
            return ONNXGraph.get_tensor_shape(self._activations_tensor_name_to_value_info[edge_name])
        self._update_activation_tensors(do_shape_inference=True)
        if edge_name in self._activations_tensor_name_to_value_info:
            return ONNXGraph.get_tensor_shape(self._activations_tensor_name_to_value_info[edge_name])
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))

    def get_edge_dtype(self, edge_name: str) -> int:
        """
        Returns the data type of the edge with the name 'edge_name'.
        If the activations tensors were not filled in self._activations_tensor_name_to_value_info, it updates them.
        If after updating of the self._activations_tensor_name_to_value_info, there is still no such tensor,
        do shape inference of the model.

        :param edge_name: The name of the edge.
        :return: Shape of the tensor on that edge.
        """
        if self._activations_tensor_name_to_value_info is None:
            self._update_activation_tensors()
        if edge_name in self._activations_tensor_name_to_value_info:
            return self._activations_tensor_name_to_value_info[edge_name].type.tensor_type.elem_type
        self._update_activation_tensors(do_shape_inference=True)
        if edge_name in self._activations_tensor_name_to_value_info:
            return self._activations_tensor_name_to_value_info[edge_name].type.tensor_type.elem_type
        raise RuntimeError('There is no edge with the name {}'.format(edge_name))

    def get_edge_dtype_name(self, edge_name: str) -> str:
        """
        Returns the name of datatype of the edge with the name 'edge_name'.

        :param edge_name: The name of the edge.
        :return: The Name of the datatype.
        """
        return onnx.TensorProto.DataType.Name(self.get_edge_dtype(edge_name))

    def get_parents(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        """
        Returns parents of the node.

        :param node: The child node.
        :return: All children nodes.
        """
        output = []
        for inp in node.input:
            output.extend(self.get_nodes_by_output(inp))
        return output

    def get_children(self, node: onnx.NodeProto) -> List[onnx.NodeProto]:
        """
        Returns children of the node.

        :param node: The parent node.
        :return: All children nodes.
        """
        output = []
        node_edges = self.get_node_edge_names(node.name)['output']
        for node_edge in node_edges:
            output.extend(self.get_nodes_by_input(node_edge))
        return output

    def is_node_shared(self, node: onnx.NodeProto) -> bool:
        """
        Returns whether the node share a weight.

        :param node: Node.
        :return: True whether node shares a weight - otherwise False.
        """
        weight_tensor_name, _ = self.get_weight_tensor(node)
        nodes = self.get_nodes_by_input(weight_tensor_name)
        return len(nodes) > 1

    def get_node_layer_name(self, node: onnx.NodeProto) -> Optional[str]:
        """
        Returns name of a weight tensor if it exists.

        :param node: Node.
        :return: Name of a weight tensor or None if the node does not have a weight.
        """
        weight_tensor_name, _ = self.get_weight_tensor(node)
        return weight_tensor_name
