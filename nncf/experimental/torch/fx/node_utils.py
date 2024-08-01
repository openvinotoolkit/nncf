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

import torch

from nncf.common.graph import NNCFNode
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.commands import TargetType
from nncf.torch.graph.transformations.commands import PTTargetPoint


# TODO(dlyakhov): Use torch.fx.graph.find_nodes method instead after
# torch version update (>= 2.4)
def get_graph_node_by_name(graph: torch.fx.Graph, name: str) -> torch.fx.Node:
    """
    Retrieves a node with the specified name from the grpah.
    Raises a runtime error if graph does not contain node with
    the given name.

    :param graph: Given torch fx graph.
    :param name: Target node name.
    :return: A graph node with the given name.
    """
    for node in graph.nodes:
        if node.name == name:
            return node
    raise RuntimeError(f"Node with name {name} is not found")


TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
    TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
    TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
}


def get_const_from_node(const_node: NNCFNode, model: torch.fx.GraphModule) -> torch.Tensor:
    """
    Retrieves a constant tensor associated with a given node.

    :param const_node: The node associated with const data.
    :param model: The NNCFNetwork object.
    :return: A torch.Tensor object containing the constant value.
    """
    assert const_node.op == "get_attr"
    target_atoms = const_node.target.split(".")
    attr_itr = model
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    const_value = attr_itr
    return const_value


def get_bias_value(node: NNCFNode, nncf_graph, model):
    bias_node = nncf_graph.get_next_nodes(node)[0]
    # TODO(dlyakhov): make a node_name_vs_node map to speed up the process
    graph_bias_node = get_graph_node_by_name(model.graph, bias_node.node_name)
    bias_value = get_const_from_node(graph_bias_node.all_input_nodes[1], model)
    bias_value = torch.flatten(bias_value)
    return bias_value


def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
    if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
        port_id = None
    if target_type in TARGET_TYPE_TO_PT_INS_TYPE_MAP:
        target_type = TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
    return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)
