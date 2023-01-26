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

import onnx
import numpy as np

from nncf.common.graph.graph import NNCFNode
from nncf.onnx.graph.metatypes.onnx_metatypes import OPERATIONS_WITH_BIAS_METATYPES
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXIdentityMetatype


def is_node_with_bias(node: NNCFNode) -> bool:
    """
    Checks if the node has a bias or not.

    :param node: The node to check.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    input_tensor_names = node.layer_attributes.input_tensor_names
    return node.metatype in OPERATIONS_WITH_BIAS_METATYPES and len(input_tensor_names) > 2


def get_bias_value(node_with_bias : NNCFNode, model: onnx.ModelProto) -> np.ndarray:
    """
    Returns the bias tensor for the biased node.

    :param node_with_bias : The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    onnx_graph = ONNXGraph(model)
    onnx_node = onnx_graph.get_node_by_name(node_with_bias.node_name)
    bias_port_id = onnx_graph.get_bias_tensor_port_id(onnx_node)
    bias_input_name = onnx_node.input[bias_port_id]
    if onnx_graph.has_initializer(bias_input_name):
        return onnx_graph.get_initializers_value(bias_input_name)
    node = onnx_graph.get_nodes_by_output(bias_input_name)[0]
    metatype = ONNX_OPERATION_METATYPES.get_operator_metatype_by_op_name(node_with_bias.op_type)
    if metatype == ONNXIdentityMetatype:
        return onnx_graph.get_initializers_value(node.input[0])
    raise RuntimeError('Could not find the bias value of the node')
