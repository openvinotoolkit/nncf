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

from typing import Dict, List, Tuple

import numpy as np
import onnx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.onnx_graph import ONNXGraph


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
