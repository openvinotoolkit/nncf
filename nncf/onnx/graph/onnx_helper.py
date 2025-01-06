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
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import onnx
from onnx import numpy_helper

import nncf


def get_name_to_node_map(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    """
    Returns mapping from node name to the node.

    :param model: Model from mapping is built.
    :return: Mapping.
    """
    return {node.name: node for node in model.graph.node}


def get_edge_info_mapping(model: onnx.ModelProto) -> Dict[str, onnx.ValueInfoProto]:
    """
    Retuns mapping from edge name to the edge info.

    :param model: Model from mapping is built.
    :return: Mapping.
    """
    return {
        tensor.name: tensor
        for tensor in (*model.graph.value_info, *model.graph.input, *model.graph.output, *model.graph.initializer)
    }


def get_children_node_mapping(model: onnx.ModelProto) -> Dict[str, List[onnx.NodeProto]]:
    """
    Returns a mapping from edge name to nodes which consume this edge as an input.

    :param model: ONNX model.
    :return: Mapping from edge name to nodes which consume this edge as an input.
    """
    output = defaultdict(list)
    for node in model.graph.node:
        for edge in node.input:
            output[edge].append(node)
    return output


def get_parents_node_mapping(model: onnx.ModelProto) -> Dict[str, onnx.NodeProto]:
    """
    Returns a mapping from edge name to node which outputs this edge.

    :param model: ONNX model.
    :return: Mapping from edge name to node which outputs this edge.
    """
    output = {}
    for node in model.graph.node:
        for edge in node.output:
            output[edge] = node
    return output


def get_model_inputs(model: onnx.ModelProto) -> List[onnx.ValueInfoProto]:
    """
    Returns all model inputs.

    :param model: ONNX model.
    :return: Model Inputs.
    """
    inputs = []
    input_all = [node.name for node in model.graph.input]
    input_initializer = [node.name for node in model.graph.initializer]
    net_feed_input = list(set(input_all) - set(input_initializer))
    for node in model.graph.input:
        if node.name in net_feed_input:
            inputs.append(node)
    return inputs


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
    raise nncf.ValidationError(f"The node {to_node} does not have input edge with the name {input_name}")


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
    raise nncf.ValidationError(f"The node {from_node} does not have output edge with the name {output_name}")


def get_node_index(model: onnx.ModelProto, node_name: str) -> Optional[int]:
    """
    Returns the node index in the model.

    :param model: ONNX model.
    :param node_name: Name of the node.
    :return: Node index, -1 if there is no such node.
    """
    for i, node in enumerate(model.graph.node):
        if node.name == node_name:
            return i
    return None


def _get_all_tensors(model: onnx.ModelProto) -> Iterator[onnx.TensorProto]:
    """
    Iterate over all tensors of ONNX model.

    :param model: ONNX model.
    :yield: tensors of ONNX model.
    """
    for initializer in model.graph.initializer:
        yield initializer
    for node in model.graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors


def has_tensor(model: onnx.ModelProto, tensor_name: str) -> bool:
    """
    Returns True whether the model has the tensor with the name equals to tensor_name.

    :param model: ONNX model.
    :param tensor_name: Name of the tensor.
    :return: True if the model has such tensor, False - otherwise.
    """
    for tensor in _get_all_tensors(model):
        if tensor.name == tensor_name:
            return True
    return False


def get_tensor(model: onnx.ModelProto, tensor_name: str) -> onnx.TensorProto:
    """
    Returns a tensor with the name 'tensor_name'.

    :param model: ONNX model.
    :param tensor_name: Name of the tensor.
    :return: The Initializer.
    """
    for tensor in _get_all_tensors(model):
        if tensor.name == tensor_name:
            return tensor
    raise nncf.ValidationError("There is no tensor with the name {}".format(tensor_name))


def get_tensor_value(model: onnx.ModelProto, tensor_name: str) -> np.ndarray:
    """
    Returns tensor value of a tensor with the name 'tensor_name'.

    :param model: ONNX model.
    :param tensor_name: Name of the tensor.
    :return: The value of the tensor.
    """
    return numpy_helper.to_array(get_tensor(model, tensor_name))


def get_edge_shape(edge: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> List[int]:
    """
    Returns edge shape.

    :param edge: The edge.
    :return: Shape of the Tensor.
    """
    if isinstance(edge, onnx.TensorProto):
        return list(edge.dims)
    tensor_type = edge.type.tensor_type
    shape = []
    if tensor_type.HasField("shape"):
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                dim_value = d.dim_value
                if isinstance(dim_value, int):
                    shape.append(dim_value)
                else:
                    return shape
            elif d.HasField("dim_param"):
                # flexible shape  make manually -1
                shape.append(-1)
            else:
                return shape
    return shape


def get_edge_dtype(edge: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> int:
    """
    Returns the data type of the edge.

    :param edge: The edge.
    :return: Data type of the edge.
    """
    if isinstance(edge, onnx.ValueInfoProto):
        return edge.type.tensor_type.elem_type
    return edge.data_type


def get_parent(
    node: onnx.NodeProto,
    port_id: int,
    parents_node_mapping: Dict[str, onnx.NodeProto],
) -> Optional[onnx.NodeProto]:
    """
    Returns parents of the node. If there is no parent node, returns None.

    :param node: The child node.
    :param port_id: Input port id on which the parent is seeked.
    :param edge_node_mapping: Mapping describing start and consumed nodes of the edges.
    :return: Parent node.
    """
    if port_id < len(node.input):
        return parents_node_mapping.get(node.input[port_id])
    return None


def get_children(node: onnx.NodeProto, children_node_mapping: Dict[str, List[onnx.NodeProto]]) -> List[onnx.NodeProto]:
    """
    Returns children of the node.

    :param node: The parent node.
    :param edge_node_mapping: Mapping describing start and consumed nodes of the edges.
    :return: All children nodes.
    """
    output = []
    for node_edge in node.output:
        output.extend(children_node_mapping[node_edge])
    return output


def is_node_has_shared_weight(
    node: onnx.NodeProto,
    weight_port_id: int,
    children_node_mapping: Dict[str, List[onnx.NodeProto]],
) -> bool:
    """
    Returns whether the node share a weight.

    :param node: Node.
    :param weight_port_id: Port id on which there is a weight.
    :param edge_node_mapping: Mapping describing start and consumed nodes of the edges.
    :return: True whether node shares a weight - otherwise False.
    """
    weight_tensor_edge = node.input[weight_port_id]
    nodes = children_node_mapping[weight_tensor_edge]
    return len(nodes) > 1
