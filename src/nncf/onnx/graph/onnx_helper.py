# Copyright (c) 2026 Intel Corporation
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
from typing import Any, Iterator, Optional, Union

import numpy as np
import onnx
from onnx import numpy_helper

import nncf
from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import get_metadata
from nncf.tensor.definitions import TensorDataType

NNCF_DTYPE_TO_ONNX_DTYPE = {
    TensorDataType.float16: onnx.TensorProto.FLOAT16,
    TensorDataType.bfloat16: onnx.TensorProto.BFLOAT16,
    TensorDataType.float32: onnx.TensorProto.FLOAT,
    TensorDataType.float64: onnx.TensorProto.DOUBLE,
    TensorDataType.int32: onnx.TensorProto.INT32,
    TensorDataType.int64: onnx.TensorProto.INT64,
    TensorDataType.int8: onnx.TensorProto.INT8,
    TensorDataType.uint8: onnx.TensorProto.UINT8,
    TensorDataType.int4: onnx.TensorProto.INT4,
    TensorDataType.uint4: onnx.TensorProto.UINT4,
}

ONNX_DTYPE_TO_NNCF_DTYPE = {v: k for k, v in NNCF_DTYPE_TO_ONNX_DTYPE.items()}


def get_name_to_node_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """
    Returns mapping from node name to the node.

    :param model: Model from mapping is built.
    :return: Mapping.
    """
    return {node.name: node for node in model.graph.node}


def get_edge_info_mapping(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    """
    Returns mapping from edge name to the edge info.

    :param model: Model from mapping is built.
    :return: Mapping.
    """
    return {
        tensor.name: tensor
        for tensor in (*model.graph.value_info, *model.graph.input, *model.graph.output, *model.graph.initializer)
    }


def get_children_node_mapping(model: onnx.ModelProto) -> dict[str, list[onnx.NodeProto]]:
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


def get_parents_node_mapping(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
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


def get_model_inputs(model: onnx.ModelProto) -> list[onnx.ValueInfoProto]:
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
    msg = f"The node {to_node} does not have input edge with the name {input_name}"
    raise nncf.ValidationError(msg)


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
    msg = f"The node {from_node} does not have output edge with the name {output_name}"
    raise nncf.ValidationError(msg)


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
    yield from model.graph.initializer
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
    msg = f"There is no tensor with the name {tensor_name}"
    raise nncf.ValidationError(msg)


def get_tensor_value(model: onnx.ModelProto, tensor_name: str) -> np.ndarray:
    """
    Returns tensor value of a tensor with the name 'tensor_name'.

    :param model: ONNX model.
    :param tensor_name: Name of the tensor.
    :return: The value of the tensor.
    """
    tensor = get_tensor(model, tensor_name)
    return get_array_from_tensor(model, tensor)


def get_array_from_tensor(model: onnx.ModelProto, tensor: onnx.TensorProto) -> np.ndarray:
    """
    Returns the data from an ONNX tensor as NumPy array.

    :param model: The ONNX model containing the tensor.
    :param tensor: The specific tensor whose data is to be extracted.
    :return: A NumPy array containing the tensor's data.
    """
    external_data_dir = get_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
    base_dir = external_data_dir if external_data_dir else ""
    return numpy_helper.to_array(tensor, base_dir)


def get_edge_shape(edge: Union[onnx.ValueInfoProto, onnx.TensorProto]) -> list[int]:
    """
    Returns edge shape.

    :param edge: The edge.
    :return: Shape of the Tensor.
    """
    if isinstance(edge, onnx.TensorProto):
        return list(edge.dims)
    tensor_type = edge.type.tensor_type
    shape = []
    if not tensor_type.HasField("shape"):
        return shape  # shape is unknown

    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            shape.append(dim.dim_value)  # Known dimension (int)
        elif dim.HasField("dim_param"):
            shape.append(-1)  # Symbolic dimension (string)
        else:
            # Unknown dimension
            # TODO(andrey-churkin): It's not clear what we should add in this case.
            # None is probably the better choice.
            shape.append(-1)

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
    parents_node_mapping: dict[str, onnx.NodeProto],
) -> Optional[onnx.NodeProto]:
    """
    Returns parents of the node. If there is no parent node, returns None.

    :param node: The child node.
    :param port_id: Input port id on which the parent is sought.
    :param edge_node_mapping: Mapping describing start and consumed nodes of the edges.
    :return: Parent node.
    """
    if port_id < len(node.input):
        return parents_node_mapping.get(node.input[port_id])
    return None


def get_children(node: onnx.NodeProto, children_node_mapping: dict[str, list[onnx.NodeProto]]) -> list[onnx.NodeProto]:
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
    children_node_mapping: dict[str, list[onnx.NodeProto]],
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


def pack_4_bits(tensor: np.ndarray) -> np.ndarray:
    """
    Apply packing based on the rule - https://onnx.ai/onnx/technical/int4.html#packing-and-unpacking
    :param tensor: Tensor to pack.
    :return: Packed tensor.
    """
    if tensor.dtype == np.uint8:
        if np.max(tensor) > 15 or np.min(tensor) < 0:
            msg = "Tensor values are not in [0, 15]."
            raise nncf.InternalError(msg)
    elif tensor.dtype == np.int8:
        if np.max(tensor) > 7 or np.min(tensor) < -8:
            msg = "Tensor values are not in [-8, 7]."
            raise nncf.InternalError(msg)
    else:
        msg = f"Invalid weight dtype {tensor.dtype}."
        raise nncf.InternalError(msg)
    packed_tensor = np.ascontiguousarray(tensor)
    packed_tensor = packed_tensor.reshape(-1, 2)
    packed_tensor = packed_tensor[..., 1::2] << 4 | packed_tensor[..., ::2] & 15
    return packed_tensor


def pack_int4_to_uint8(weight: np.ndarray, block_size: int, signed: bool) -> np.ndarray:
    """
    Returns `weight` that is stored as uint8 with shape (N, n_blocks_per_col, blob_size) in which:
        - n_blocks_per_col = CeilDiv(K, block_size)
        - blob_size = CeilDiv(block_size * bits, 8)
        - bits = 4 (Number of bits used for weight quantization)

    See https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits
    for more details.

    :param weight: A 2D array of shape (K, N) quantized with 4 bits.
    :param block_size: Number of groupsize used for weight quantization.
    :param signed: True if the weight has type int4, and False if uint4.
    :return: A packed weight that can be used as `B` input for `com.microsoft.MatMulNBits` operation.
    """
    ceil_div = lambda a, b: (a + b - 1) // b
    K, N = weight.shape
    n_blocks_per_col = ceil_div(K, block_size)

    if signed:
        if weight.dtype != np.int8:
            msg = f"Expected weight dtype to be np.int8 for signed weight tensor, but got {weight.dtype}"
            raise nncf.ValidationError(msg)
        weight = weight + 8  # [-8, 7] -> [0, 15]
        weight = weight.astype(np.uint8)

    K_padded = n_blocks_per_col * block_size
    pad_len = K_padded - K
    if pad_len > 0:
        weight = np.pad(weight, ((0, pad_len), (0, 0)), mode="constant", constant_values=0)

    weight_blocks = weight.reshape(n_blocks_per_col, block_size, N)

    even = weight_blocks[:, 0::2, :]
    odd = weight_blocks[:, 1::2, :]
    if odd.shape[1] < even.shape[1]:
        pad_width = [(0, 0), (0, even.shape[1] - odd.shape[1]), (0, 0)]
        odd = np.pad(odd, pad_width, mode="constant", constant_values=0)

    packed = ((odd & 0x0F) << 4) | (even & 0x0F)  # (n_blocks_per_col, blob_size, N)
    packed_weight = packed.transpose(2, 0, 1)

    return packed_weight


def get_node_attr_value(node: onnx.NodeProto, attr_name: str) -> Optional[Any]:
    """
    Retrieves the value of a specified attribute from a node.

    This function searches for an attribute with the given name in the provided
    node. If the attribute exists, its value is returned. If the attribute is
    not found, `None` is returned. If multiple attributes with the same name are
    found, a `ValueError` is raised.

    :param node: The node to retrieve the attribute from.
    :param attr_name: The name of the attribute to retrieve.
    :return: The value of the attribute if found; otherwise, `None`.
    """
    matching = [x for x in node.attribute if x.name == attr_name]

    if len(matching) > 1:
        msg = f"Node has multiple attributes with name {attr_name}."
        raise ValueError(msg)

    if len(matching) < 1:
        return None

    return onnx.helper.get_attribute_value(matching[0])
