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

from typing import Optional

import numpy as np
import openvino.runtime as ov

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvertMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConstantMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OPERATIONS_WITH_BIAS_METATYPES


def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Checks if the node has a bias or not.

    :param node: The node to check.
    :param nncf_graph: NNCFGraph instance.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    if node.metatype not in OPERATIONS_WITH_BIAS_METATYPES:
        return False

    add_node = nncf_graph.get_next_nodes(node)[0]
    if add_node.metatype != OVAddMetatype:
        return False

    bias_constant = get_node_with_bias_value(add_node, nncf_graph)
    return bias_constant is not None


def get_const_value(const_node: ov.Node) -> np.ndarray:
    """
    Returns the constant tensor for the node.

    :param const_node: OpenVINO node.
    :return: The constant value.
    """
    return const_node.get_vector().reshape(const_node.get_output_shape(0))


def get_bias_value(node_with_bias: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
    """
    Returns the bias tensor for the biased node.

    :param node_with_bias: The node that corresponds to the operation with bias.
    :param nncf_graph: NNCFGraph instance.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    ops_dict = {op.get_friendly_name(): op for op in model.get_ops()}

    add_node = nncf_graph.get_next_nodes(node_with_bias)[0]
    bias_constant = get_node_with_bias_value(add_node, nncf_graph)
    ov_bias_constant = ops_dict[bias_constant.node_name]
    return get_const_value(ov_bias_constant)


def get_weight_value(node_with_weight: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model, port_id: int) -> np.ndarray:
    """
    Returns a weight value for the node with weight.

    :param node_with_weight: Node with weight.
    :param nncf_graph: NNCF graph.
    :param model: The model that contains this operation.
    :param port_id: The input port ID to get weight input.
    :return: The weight value.
    """
    node = nncf_graph.get_input_edges(node_with_weight)[port_id].from_node
    if node.metatype == OVConvertMetatype:
        node = nncf_graph.get_input_edges(node)[0].from_node

    friendly_name_to_op_map = {op.get_friendly_name(): op for op in model.get_ops()}
    const_op = friendly_name_to_op_map[node.node_name]
    weight_tensor = get_const_value(const_op)
    return weight_tensor


def get_node_with_bias_value(add_node: NNCFNode, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Returns node that represents bias constant in the NNCF graph, if it exists.

    :param add_node: NNCFNode that provides bias.
    :param nncf_graph: NNCFGraph instance.
    :return: Optional NNCFNode with bias value.
    """
    if add_node.layer_attributes is None:
        return None

    const_port_ids = add_node.layer_attributes.get_const_port_ids()
    assert len(const_port_ids) == 1
    bias_port_id = const_port_ids[0]
    bias_constant = nncf_graph.get_input_edges(add_node)[bias_port_id].from_node

    if bias_constant.metatype == OVConvertMetatype:
        bias_constant = nncf_graph.get_input_edges(bias_constant)[0].from_node

    return bias_constant if bias_constant.metatype == OVConstantMetatype else None


def get_result_node_name(output_name: str, port_id: int) -> str:
    """
    Returns name of Result based on node name and its port.

    :param output_name: Node name.
    :param port_id: Node port.
    :return: Name of result.
    """

    return f'Result_{output_name}.{port_id}'
