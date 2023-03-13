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


def get_result_node_name(output_name: str, port_id: int) -> str:
    """
    Returns name of Result based on node name and its port.

    :param output_name: Node name.
    :param port_id: Node port.
    :return: Name of result.
    """

    return f'Result_{output_name}.{port_id}'


def get_reduce_node_name(output_name: str, node_type: str) -> str:
    return f'{output_name}_{node_type}'


def get_output_edge_name(edge_name: str, reduce_type: str):
    return f'Edge_{edge_name}_{reduce_type}'


def get_inplace_reduce_op(op_type, reduction_axes, use_abs):
    def get_reduce_op(node_name, op_input_edge):
        ops = []
        if use_abs:
            abs_node_type = 'Abs'
            reduce_input_edge = get_output_edge_name(op_input_edge, abs_node_type)
            op_input = onnx.helper.make_node(
                name=get_reduce_node_name(node_name, abs_node_type),
                op_type=abs_node_type,
                inputs=[op_input_edge],
                outputs=[reduce_input_edge],
            )
            ops.append(op_input)
        else:
            reduce_input_edge = op_input_edge

        reduce_op = onnx.helper.make_node(
            op_type=op_type,
            axes=reduction_axes,
            keep_dims=True,
            inputs=[reduce_input_edge],
            outputs=[get_output_edge_name(op_input_edge, op_type)],
            name=get_reduce_node_name(node_name, op_type))
        ops.append(reduce_op)
        return ops
    return get_reduce_op


def get_inplace_min_op(reduction_shape):
    return get_inplace_reduce_op('ReduceMin', reduction_shape, False)


def get_inplace_max_op(reduction_shape, use_abs_max):
    return get_inplace_reduce_op('ReduceMax', reduction_shape, use_abs_max)

def get_inplace_mean_op(reduction_shape):
    return get_inplace_reduce_op('ReduceMean', reduction_shape, False)
