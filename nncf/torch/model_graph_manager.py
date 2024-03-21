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

from typing import List, Optional, Tuple, Union

import torch

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import CONST_NOOP_METATYPES
from nncf.torch.dynamic_graph.context import PreHookId
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.external_quantizer import ExternalQuantizerCallHook
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer

CONV_META_TYPES = [
    om.PTConv1dMetatype,
    om.PTConv2dMetatype,
    om.PTConv3dMetatype,
    om.PTDepthwiseConv1dSubtype,
    om.PTDepthwiseConv2dSubtype,
    om.PTDepthwiseConv3dSubtype,
    om.PTConvTranspose1dMetatype,
    om.PTConvTranspose2dMetatype,
    om.PTConvTranspose3dMetatype,
]

OPERATORS_WITH_BIAS_METATYPES = CONV_META_TYPES
CONV_FUSED_META_TYPES = [om.PTBatchNormMetatype]


def find_const_node_in_constant_subgraph(node: NNCFNode, graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Finds a constant node within a constant subgraph, recursively traversing noop and quantize nodes.

    :param node: The starting node to search from.
    :param graph: The NNCFGraph.
    :return: The constant node found within the subgraph, or None if no constant node is found.
    """
    if node.metatype == om.PTNoopMetatype or node.node_type in om.QUANTIZE_NODE_TYPES:
        prev_nodes = graph.get_previous_nodes(node)
        if len(prev_nodes) != 1:
            return None
        return find_const_node_in_constant_subgraph(prev_nodes[0], graph)
    if node.metatype in CONST_NOOP_METATYPES:
        return node
    return None


def get_const_node(node: NNCFNode, port_id: int, graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Retrieves the constant node providing the input to a specific port of a given node in the NNCF graph.

    :param node: The NNCF node for which to find the constant input node.
    :param port_id: The ID of the input port to consider.
    :param graph: The NNCF graph containing the nodes.
    :return: The NNCF node providing the constant input to the specified port, or None if no such node is found.
    """
    for prev_node in graph.get_previous_nodes(node):
        edge = graph.get_edge(prev_node, node)
        if edge.input_port_id == port_id:
            weight_node = find_const_node_in_constant_subgraph(prev_node, graph)
            if weight_node is None:
                raise nncf.InternalError("Could not find a constant node in the model graph.")
            return weight_node


def split_const_name(const_name: str) -> Tuple[str, str]:
    """
    Splits the constant name into module and attribute names.

    :param const_name: The full name of the constant, including module and attribute.
    :return:
        - module_name: The name of the module containing the constant.
        - weight_attr_name: The name of the constant attribute within the module.
    """
    index = const_name.rfind(".")
    if index == -1:
        return str(), const_name
    module_name = const_name[:index]
    weight_attr_name = const_name[index + 1 :]
    return module_name, weight_attr_name


def get_module_by_name(module_name: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Retrieves a module from a PyTorch model by its hierarchical name.

    :param module_name: The name of the module to retrieve (e.g., "module1.submodule2").
    :param model: The model to search within.
    :return: The retrieved module.
    """
    if not module_name:
        return model
    curr_module = model
    for name in module_name.split("."):
        for child_name, child_module in curr_module.named_children():
            if child_name == name:
                curr_module = child_module
                break
        else:
            raise nncf.ModuleNotFoundError(f"Could not find the {module_name} module in the model.")
    return curr_module


def get_const_data(const_node: NNCFNode, model: NNCFNetwork) -> torch.Tensor:
    """
    Retrieves a constant tensor associated with a given node.

    :param const_node: The node associated with const data.
    :param model: The NNCFNetwork object.
    :return: A torch.Tensor object containing the constant value.
    """
    const_name = const_node.layer_attributes.name
    module_name, const_attr_name = split_const_name(const_name)
    module = get_module_by_name(module_name, model)
    data = getattr(module, const_attr_name)
    if isinstance(data, torch.nn.Parameter):
        return data.data
    return data


def get_const_data_on_port(node: NNCFNode, port_id: int, model: NNCFNetwork) -> torch.Tensor:
    """
    Retrieves a constant tensor associated with a given node and input port in an NNCF graph.

    :param node: The node to retrieve the constant from.
    :param port_id:  The port id within the node that holds the constant.
    :param model: The NNCFNetwork object.
    :return: A torch.Tensor object containing the constant value, or None if the constant is not found.
    """
    graph = model.nncf.get_graph()
    const_node = get_const_node(node, port_id, graph)
    if const_node is None:
        return None
    return get_const_data(const_node, model)


def get_potential_fused_node(node_name: str, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
    """
    Retrieves the next node in the NNCF graph that could be fused with the provided node during runtime optimization.

    :param node_name: The node name.
    :param nncf_graph: The NNCF graph.
    :return: The node that can be fused or None if no suitable node is found.
    """
    target_node = nncf_graph.get_node_by_name(node_name)

    if target_node.metatype in CONV_META_TYPES:
        next_nodes = nncf_graph.get_next_nodes(target_node)
        for node in next_nodes:
            if node.metatype in CONV_FUSED_META_TYPES:
                return node
    return None


def is_node_with_fused_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Checks if the node has a fused bias.

    :param node: The node to check.
    :param nncf_graph: The NNCF graph.
    :return: Return `True` if `node` corresponds to the operation
        with bias (bias is added to the output tensor of that operation),
        `False` otherwise.
    """
    if node.metatype not in OPERATORS_WITH_BIAS_METATYPES:
        return False
    fused_node = get_potential_fused_node(node.node_name, nncf_graph)
    if fused_node is not None:
        node = fused_node
    bias_port = node.metatype.bias_port_id
    bias = get_const_node(node, bias_port, nncf_graph)
    return bias is not None


def get_fused_bias_value(node: NNCFNode, model: NNCFNetwork) -> Optional[torch.Tensor]:
    """
    Returns the bias tensor for the node or for potential fused node.

    :param node: The node that corresponds to the operation with bias.
    :param model: The model that contains this operation.
    :return: The bias value that is applied to the output tensor of the node's operation.
    """
    nncf_graph = model.nncf.get_graph()
    fused_node = get_potential_fused_node(node.node_name, nncf_graph)
    target_node_name = fused_node.node_name if fused_node else node.node_name
    target_node = nncf_graph.get_node_by_name(target_node_name)
    return get_const_data_on_port(target_node, target_node.metatype.bias_port_id, model)


def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[int]:
    """
    Returns list of input port ids that contains traced constant tensor.

    :param node: Target node that contains weights.
    :param graph: The NNCF graph.
    :return: List of ports with weights.
    """
    weight_port_ids = []
    for edge in graph.get_input_edges(node):
        if edge.input_port_id in node.metatype.weight_port_ids:
            weight_node = find_const_node_in_constant_subgraph(edge.from_node, graph)
            if weight_node:
                weight_port_ids.append(edge.input_port_id)
    return weight_port_ids


def set_const_data(data: torch.Tensor, const_node: NNCFNode, model: NNCFNetwork) -> None:
    """
    Sets the constant data associated with a specific constant node in an NNCF network model.

    :param data: The constant data tensor to be set.
    :param const_node: The NNCF node representing the constant data.
    :param model: The NNCF network model.
    """
    const_name = const_node.layer_attributes.name
    module_name, const_attr_name = split_const_name(const_name)
    module = get_module_by_name(module_name, model)
    const = getattr(module, const_attr_name)
    if isinstance(const, torch.nn.Parameter):
        const.data = data
    else:
        setattr(module, const_attr_name, data)


def set_const_data_to_port_id(data: torch.Tensor, node: NNCFNode, port_id: int, model: NNCFNetwork) -> None:
    """
    Sets the value of a constant tensor within a specified node in an NNCFNetwork.

    :param data: The tensor containing the new value to be set for the constant.
    :param node: The NNCF node representing the operation that uses the constant.
    :param const_port_id: The input port id of the node that receives the constant.
    :param model: The NNCF network containing the module to be modified.
    """
    graph = model.nncf.get_graph()
    const_node = get_const_node(node, port_id, graph)
    if const_node is None:
        raise nncf.InternalError(f"No found node with constant for {node.node_name} on {port_id} port")
    const_name = const_node.layer_attributes.name
    module_name, const_attr_name = split_const_name(const_name)
    module = get_module_by_name(module_name, model)
    const = getattr(module, const_attr_name)
    if isinstance(const, torch.nn.Parameter):
        const.data = data
    else:
        setattr(module, const_attr_name, data)


def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
    """
    Check that module have fake_quantizer for its weights (supports only metatypes with only one weight port).

    :param node: The target node.
    :param nncf_graph: The NNCF graph.
    :return bool: return `True` if the node is quantized.
    """
    assert len(node.metatype.weight_port_ids) == 1, "Support only metatype with only 1 weighted port"
    for edge in nncf_graph.get_input_edges(node):
        if edge.input_port_id in node.metatype.weight_port_ids and edge.from_node.node_type in om.QUANTIZE_NODE_TYPES:
            return True
    return False


def get_fake_quantizer(
    node: NNCFNode, port_id: Optional[int], model: NNCFNetwork
) -> Union[SymmetricQuantizer, AsymmetricQuantizer]:
    """
    Retrieves the fake quantizer associated with a specific node and input port id.

    :param node: The NNCFNode representing the node for which to retrieve the quantizer.
    :param port_id: The port id number for which to retrieve the quantizer module, None means output port.
    :param model: The NNCFNetwork instance.
    :return: Fake Quantizer module if exists, overwise None.
    """

    address_map = model.nncf.get_node_to_op_address_mapping()
    op_addr = address_map[node.node_name]

    if port_id is not None:
        id = PreHookId(op_address=op_addr, input_port_id=port_id)
        hook_container = model.nncf._compressed_context._pre_hooks.get(id, {})
    else:
        hook_container = model.nncf._compressed_context._post_hooks.get(op_addr, {})

    for call_hook in hook_container.values():
        if isinstance(call_hook, ExternalQuantizerCallHook):
            storage = getattr(model.nncf, call_hook._storage_name)
            return storage[call_hook._storage_key]
    return None
