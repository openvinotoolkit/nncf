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

from typing import Dict, List, Optional, Tuple

import numpy as np
import onnx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.logging.logger import nncf_logger
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.onnx.graph.metatypes import onnx_metatypes as om
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint


def is_node_with_bias(node: NNCFNode) -> bool:
    """
    Checks if the node has a bias or not.

    :param node: The node to check.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    return node.layer_attributes.has_bias()


def get_bias_value(node_with_bias: NNCFNode, model: onnx.ModelProto) -> np.ndarray:
    """
    Returns the bias tensor for the biased node.

    :param node_with_bias: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    onnx_graph = ONNXGraph(model)
    assert node_with_bias.layer_attributes.has_bias()
    bias_name = node_with_bias.layer_attributes.bias_attrs["name"]
    return onnx_graph.get_tensor_value(bias_name)


def get_input_edges_mapping(nncf_graph: NNCFGraph) -> Dict[str, Tuple[str, int]]:
    """
    Returns mapping between NNCFGraph input nodes and following by ONNX nodes with corresponding input port ids.

    :param nncf_graph: instance of NNCFGraph
    :return: A mapping of NNCF input node names and a tuple with the consumed node names and their input port ids.
    """
    input_edges_mapping = {}
    for input_node in nncf_graph.get_input_nodes():
        input_edges_mapping[input_node.node_name] = []
        for next_node in nncf_graph.get_next_nodes(input_node):
            for edge in nncf_graph.get_input_edges(next_node):
                if edge.from_node == input_node:
                    input_edges_mapping[input_node.node_name].append((next_node.node_name, edge.input_port_id))
    return input_edges_mapping


def get_input_edge(input_node_name: str, input_edges_mapping: Dict[str, Tuple[str, int]], onnx_graph: ONNXGraph) -> str:
    """
    Returns input edge corresponding to the NNCF input node with the name input_node_name.

    :param input_node_name: Name of NNCF input node.
    :param input_edges_mapping: A mapping of NNCF input node names and
        a tuple with the consumed node names and their input port ids.
    :param onnx_graph: Instance of ONNXGraph of the model.
    :return: Input edge name.
    """
    input_edges = set()
    for node_info in input_edges_mapping[input_node_name]:
        name, port_id = node_info
        input_edges.add(onnx_graph.get_node_edge_names(name)["input"][port_id])
    assert len(input_edges) == 1
    return input_edges.pop()


def is_any_weight_quantized(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Returns True if any weight port id of node is quantized,
    False - if all weights are not quantized or the node can not have weight.

    :param node: NNCFNode.
    :param nncf_graph: NNCGraph.
    :return: True if any weight port id of node is quantized,
    False - if all weights are not quantized or the node can not have weight.
    """
    is_quantized_weight = False
    if node.layer_attributes.has_weight():
        for port_id in node.layer_attributes.weight_attrs.keys():
            is_quantized_weight = is_quantized_weight or is_port_quantized(node, nncf_graph, port_id)
    return is_quantized_weight


def is_port_quantized(node: NNCFNode, nncf_graph: NNCFGraph, port_id: int) -> bool:
    """
    Returns True if a port_id is quantized - have ONNXDequantizeLinearMetatype as a parent node.

    :param node: NNCFNode.
    :param nncf_graph: NNCFGraph.
    :param port_id: Input port id of a node.
    :return: True if a port_id is quantized - have ONNXDequantizeLinearMetatype as a parent node.
    """
    input_nodes = [edge.from_node for edge in nncf_graph.get_input_edges(node)]
    if len(input_nodes) > port_id:
        weight_node = input_nodes[port_id]
        return weight_node.metatype == ONNXDequantizeLinearMetatype
    return False


def transpose_axis(shape: List[int], axis: int) -> int:
    """
    Returns transpose axis.

    :param shape: Tensor shape.
    :param axis: Axis before transpose.
    :return: Axis after transpose.
    """
    axis %= len(shape)  # Make axis positive
    return range(len(shape) - 1, -1, -1)[axis]  # Iterate backward throug axis


def get_reduction_shape(shape: List[int], axis: int) -> ReductionShape:
    """
    Returns reduction shape for shape and axis.

    :param shape: Shape.
    :param axis: Axis.
    :return: Reduction shape.
    """
    reduction_shape = list(range(len(shape)))
    if len(reduction_shape) == 1:  # If only one channel
        return tuple(reduction_shape)
    reduction_shape.pop(axis)
    return tuple(reduction_shape)


def _get_weight_quantization_axis(node: NNCFNode, port_id: int) -> int:
    """
    Returns weight tensor axis along quantizer parameters are calculated.

    :param node: NNCFNode, which has a weight on input port_id.
    :param port_id: Input port id on which there is a weight of a node.
    :return: Axis along quantizer parameters are calculated.
    """
    weight_channel_axis = node.metatype.weight_channel_axis
    if node.layer_attributes.has_node_attrs():
        if node.metatype == om.ONNXGemmMetatype:
            weight_shape = node.layer_attributes.weight_attrs[port_id]["shape"]
            if (
                port_id == 0
                and node.layer_attributes.node_attrs["transA"] == 1
                or port_id == 1
                and node.layer_attributes.node_attrs["transB"] == 1
            ):
                weight_channel_axis = transpose_axis(weight_shape, weight_channel_axis)
    return weight_channel_axis


def _get_activation_quantization_axis() -> int:
    """
    Returns activation tensor axis along quantizer parameters are calculated.

    :return: Axis along quantizer parameters are calculated.
    """
    return 1  # Activations have channel first layout: [N, C, Z, Y, X]


def _get_activation_tensor_shape(
    nncf_graph: NNCFGraph, node: NNCFNode, target_point: ONNXTargetPoint
) -> Optional[List[int]]:
    """
    Returns shape of an activation tensor which is correspond to the target point and node.
    ONNX model can not have a shape of a edge, even after shape inference.
    Therefore, if there is no info regarding shape, None is returned.

    :param nncf_graph: NNCFGraph.
    :param node: NNCFNode.
    :param target_point: Determines from input or ouput of a node take a shape info.
    :return: None, if there is no shape info, otherwise - tensor shape.
    """
    if target_point.type == TargetType.PRE_LAYER_OPERATION:
        shape = nncf_graph.get_input_edges(node)[target_point.port_id].tensor_shape
    elif target_point.type == TargetType.POST_LAYER_OPERATION:
        shape = nncf_graph.get_output_edges(node)[target_point.port_id].tensor_shape
    else:
        raise NotImplementedError(f"Unsupported target point type {target_point.type}.")
    if not shape:  # ONNX model can not have a shape of a edge, even after shape inference.
        if target_point.type == TargetType.PRE_LAYER_OPERATION:
            nncf_logger.info(
                f"The shape of input edge of a node {node.node_name} is unkown. \
                    Therefore per-tensor quantizaiton is applied."
            )
        elif target_point.type == TargetType.POST_LAYER_OPERATION:
            nncf_logger.info(
                f"The shape of output edge of a node {node.node_name} is unkown. \
                    Therefore per-tensor quantizaiton is applied."
            )
        nncf_logger.info("Please consider to run pre-processing before quantization.")
        # TODO: add preprocessing tool for ONNX model.
        return None
    return shape


def get_quantized_tensor_shape(
    nncf_graph: NNCFGraph, node: NNCFNode, target_point: ONNXTargetPoint
) -> Optional[List[int]]:
    """
    Returns quantized tensor shape corresponding to a target point with a node if shape - info is existed.
    If there is no shape info - returns None.

    :param nncf_graph: NNCFGraph.
    :param node: NNCFNode.
    :param target_point: Target point indicates the quantizer place in the model graph.
    :return: Shape of a quantized tensor, if shape is existed. Otherwise - None.
    """
    if target_point.is_weight_target_point():
        return node.layer_attributes.weight_attrs[target_point.port_id]["shape"]
    return _get_activation_tensor_shape(nncf_graph, node, target_point)


def get_quantization_axis(is_per_channel: bool, node: NNCFNode, target_point: ONNXTargetPoint) -> Optional[int]:
    """
    Returns axis of quantizer parameters are calculated along.
    If quantization is per-tensor returns None.

    :param is_per_channel: True if quantizater is per-channel.
    :param node: NNCFNode.
    :param target_point: Target point indicates the quantizer place in the model graph.
    :return: None if per-tensor, otherwise quantizion axis.
    """
    if not is_per_channel:
        return None
    if target_point.is_weight_target_point():
        return _get_weight_quantization_axis(node, target_point.port_id)
    return _get_activation_quantization_axis()
