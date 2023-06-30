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

from typing import Dict, Tuple

import numpy as np
import onnx

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import OPERATIONS_WITH_BIAS_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXIdentityMetatype
from nncf.onnx.graph.nncf_graph_builder import ONNXExtendedLayerAttributes
from nncf.onnx.graph.onnx_graph import ONNXGraph


def is_node_with_bias(node: NNCFNode) -> bool:
    """
    Checks if the node has a bias or not.

    :param node: The node to check.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    if node.metatype in OPERATIONS_WITH_BIAS_METATYPES and isinstance(
        node.layer_attributes, ONNXExtendedLayerAttributes
    ):
        return len(node.layer_attributes.input_tensor_names) > 2
    return False


def get_bias_value(node_with_bias: NNCFNode, model: onnx.ModelProto) -> np.ndarray:
    """
    Returns the bias tensor for the biased node.

    :param node_with_bias: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    onnx_graph = ONNXGraph(model)
    onnx_node = onnx_graph.get_node_by_name(node_with_bias.node_name)
    bias_port_id = onnx_graph.get_bias_tensor_port_id(onnx_node)
    bias_input_name = onnx_node.input[bias_port_id]
    if onnx_graph.has_tensor(bias_input_name):
        return onnx_graph.get_tensor_value(bias_input_name)
    node = onnx_graph.get_nodes_by_output(bias_input_name)[0]
    metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node.op_type)
    if metatype == ONNXIdentityMetatype:
        return onnx_graph.get_tensor_value(node.input[0])
    raise RuntimeError("Could not find the bias value of the node")


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
