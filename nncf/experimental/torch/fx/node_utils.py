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

import torch.fx

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.experimental.torch.fx.groups import FX_OPERATORS_WITH_BIAS_METATYPES
from nncf.tensor import Tensor


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


def get_tensor_constant_from_node(constant_node: torch.fx.Node, model: torch.fx.GraphModule) -> torch.nn.Parameter:
    """
    Retrieves tensor from the given constant node.

    :param constant_node: Given constant node.
    :param model: Given model.
    :return: Torch tensor referenced by the given constant node.
    """
    if constant_node is None:
        return None
    if constant_node.op != "get_attr":
        raise RuntimeError(f"Given node op == {constant_node.op}, but get_attr is expected.")
    target_atoms = constant_node.target.split(".")
    attr_itr = model
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr


def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Returns True if the node has a bias, False otherwise.

    :param node: Target node.
    :param nncf_graph: Target nncf_graph.
    :return: True if the node has a bias, False otherwise.
    """
    # Assumes that all biases were unfused
    if node.metatype not in FX_OPERATORS_WITH_BIAS_METATYPES or len(nncf_graph.get_input_edges(node)) != 3:
        return False
    const_node = nncf_graph.get_input_edge_by_port_id(node, 2).from_node
    return const_node.metatype is om.PTConstNoopMetatype


def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: torch.fx.GraphModule) -> Tensor:
    """
    Retrieves the bias value from the given node.

    :param node: Target node.
    :param nncf_graph: Target nncf_graph.
    :param model: Target GraphModule.
    :return: Bias value of the given node.
    """
    bias_node = nncf_graph.get_input_edge_by_port_id(node, 2).from_node
    # TODO(dlyakhov): make a node_name_vs_node map to speed up the process
    graph_bias_const = get_graph_node_by_name(model.graph, bias_node.node_name)
    return Tensor(get_tensor_constant_from_node(graph_bias_const, model))
