# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import openvino as ov

from nncf.openvino.graph.metatypes.groups import ELEMENTWISE_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVFakeQuantizeMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_node_metatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_operation_const_op


def get_ops_by_metatypes(model: ov.Model, metatypes: List[OVOpMetatype]):
    ops = []
    for op in model.get_ops():
        if get_node_metatype(op) in metatypes:
            ops.append(op)
    return ops


def get_ops_with_input_constants(elementwise_nodes):
    filtered_nodes = []
    for node in elementwise_nodes:
        port_0_const_op = get_operation_const_op(node, 0)
        port_1_const_op = get_operation_const_op(node, 1)
        if port_0_const_op is None and port_1_const_op is None:
            continue
        filtered_nodes.append(node)
    return filtered_nodes


def get_ops_with_FQ_activation_inputs(elementwise_nodes):
    fq_nodes = []
    for node in elementwise_nodes:
        port_0_const_op = get_operation_const_op(node, 0)
        port_1_const_op = get_operation_const_op(node, 1)
        constant_port, activation_port = -1, -1
        if port_0_const_op:
            constant_port, activation_port = 0, 1
        elif port_1_const_op:
            constant_port, activation_port = 1, 0
        assert constant_port in [0, 1] and activation_port in [0, 1]
        activation_port_input_node = node.input_value(activation_port).get_node()
        if get_node_metatype(activation_port_input_node) == OVFakeQuantizeMetatype:
            fq_nodes.append(activation_port_input_node)

    return fq_nodes


def get_fq_before_elementwise(model):
    elementwise_nodes = get_ops_by_metatypes(model, ELEMENTWISE_OPERATIONS)
    elementwise_nodes_with_constant = get_ops_with_input_constants(elementwise_nodes)
    fqs = get_ops_with_FQ_activation_inputs(elementwise_nodes_with_constant)
    return fqs
