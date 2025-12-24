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
from typing import Any, Callable, Optional, Sequence

import torch
from torch import nn

import nncf
from nncf import nncf_logger
from nncf.common.graph.graph import NNCFNode
from nncf.torch.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_data_on_port
from nncf.torch.model_graph_manager import get_const_node

CONV_METATYPES = (
    om.PTConv1dMetatype,
    om.PTConv2dMetatype,
    om.PTConv3dMetatype,
    om.PTDepthwiseConv1dSubtype,
    om.PTDepthwiseConv2dSubtype,
    om.PTDepthwiseConv3dSubtype,
)


class ExtractedFunc(nn.Module):
    """
    Module to execute function with kwargs.
    Support function only with one input.

    :param fn: Function to execute.
    :param kwargs: Function arguments.
    """

    def __init__(self, fn: Callable[..., torch.Tensor], kwargs: dict[str, Any]) -> None:
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x, **self.kwargs)


def apply_args_to_kwargs(
    args: Sequence[Any], kwargs: dict[str, Any], indexed_args: list[tuple[int, str]]
) -> dict[str, Any]:
    """
    Applies the given arguments and keyword arguments to a dictionary of keyword arguments.

    :param args: The positional arguments.
    :param kwargs: The keyword arguments.
    :param indexed_args: The list of pairs of indexes and names.
    :return: A dictionary of keyword arguments with the applied arguments and keyword arguments.
    """
    args_dict: dict[str, Any] = dict()
    for idx, arg_name in indexed_args:
        if idx < len(args):
            args_dict[arg_name] = args[idx]
        elif arg_name in kwargs:
            args_dict[arg_name] = kwargs[arg_name]

    return args_dict


def extract_bn(model: nn.Module, graph: PTNNCFGraph, node: NNCFNode) -> ExtractedFunc:
    """
    Extract batch_norm operation.

    :param model: Source model.
    :param graph: Graph of source model.
    :param node: Target batch_norm node.
    :return: BatchNorm module with same attributes and parameters from source module or None.
    """
    layer_attr = node.layer_attributes
    if not isinstance(layer_attr, PT2OpLayerAttributes):
        msg = f"Expected PT2OpLayerAttributes for input_node.layer_attributes, actual: {type(layer_attr)}"
        raise nncf.InternalError(msg)

    # torch.batch_norm(
    #   0 - input: Tensor,
    #   1 - weight: Optional[Tensor]
    #   2 - bias: Optional[Tensor]
    #   3 - running_mean: Optional[Tensor]
    #   4 - running_var: Optional[Tensor]
    #   5 - training: _bool
    #   6 - momentum: _float
    #   7 - eps: _float
    #   8 - cudnn_enabled: _bool
    # ) -> Tensor: ...

    weight = get_const_data_on_port(model, graph, node, 1)
    bias = get_const_data_on_port(model, graph, node, 2)
    running_mean = get_const_data_on_port(model, graph, node, 3)
    running_var = get_const_data_on_port(model, graph, node, 4)

    bn_kwargs = apply_args_to_kwargs(
        layer_attr.op_args,
        layer_attr.op_kwargs,
        [(6, "momentum"), (7, "eps"), (8, "cudnn_enabled")],
    )
    bn_kwargs["weight"] = weight
    bn_kwargs["bias"] = bias
    bn_kwargs["running_mean"] = running_mean
    bn_kwargs["running_var"] = running_var
    bn_kwargs["training"] = False

    return ExtractedFunc(layer_attr.func, bn_kwargs)


def extract_conv(
    model: nn.Module,
    graph: PTNNCFGraph,
    input_node: NNCFNode,
    output_node: NNCFNode,
) -> nn.Module:
    """
    Extracts a convolutional layer from an NNCF graph and constructs an ExtractedFunc module.

    :param model: The nn.Module containing the layer.
    :param graph: The NNCF graph.
    :param input_nodes: The name of input node.
    :param output_nodes: The name of output node.
    :return: The extracted convolutional layer as an ExtractedFunc module.
    """
    # torch.conv*d(
    #   0 - input: Tensor
    #   1 - weight: Tensor
    #   2 - bias: Optional[Tensor]
    #   3 - stride: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]]
    #   4 - padding: Union[Union[_int, SymInt] | str
    #   5 - dilation: Union[Union[_int, SymInt], Sequence[Union[_int, SymInt]]]
    #   6 - groups: Union[_int, SymInt]
    # ) -> Tensor: ...

    weight_node = get_const_node(input_node, 1, graph)
    if weight_node is None:
        msg = f"Weight node not found for {input_node}"
        raise nncf.InternalError(msg)
    weight = get_const_data(weight_node, model)

    hook_storage = get_hook_storage(model)
    with torch.no_grad():
        # Calculate weight after execution all hook for weight data
        weight = hook_storage.execute_post_function_hooks(weight_node.node_name, 0, weight)
        weight = hook_storage.execute_pre_function_hooks(input_node.node_name, 1, weight)

    bias_node = get_const_node(input_node, 2, graph)
    bias = get_const_data(bias_node, model) if bias_node is not None else None

    layer_attrs = input_node.layer_attributes

    if not isinstance(layer_attrs, PT2OpLayerAttributes):
        msg = f"Expected PT2OpLayerAttributes for input_node.layer_attributes, actual: {type(layer_attrs)}"
        raise nncf.InternalError(msg)

    conv_kwargs = apply_args_to_kwargs(
        layer_attrs.op_args,
        layer_attrs.op_kwargs,
        [(3, "stride"), (4, "padding"), (5, "dilation"), (6, "groups")],
    )
    conv_kwargs["weight"] = weight
    conv_kwargs["bias"] = bias
    conv_module = ExtractedFunc(layer_attrs.func, conv_kwargs)

    if input_node == output_node:
        return conv_module

    if output_node.metatype is not om.PTBatchNormMetatype:
        msg = f"Support only PTBatchNormMetatype as output node, actual: {output_node.metatype}"
        raise nncf.InternalError(msg)

    next_nodes = graph.get_next_nodes(input_node)
    if output_node not in next_nodes:
        msg = f"Output node {output_node} not found after {input_node}"
        raise nncf.InternalError(msg)

    bn_module = extract_bn(model, graph, output_node)
    return nn.Sequential(conv_module, bn_module)


def extract_linear(
    model: nn.Module,
    graph: PTNNCFGraph,
    input_node: NNCFNode,
    output_node: NNCFNode,
) -> Optional[nn.Module]:
    """
    Extracts a linear layer from an NNCF graph and constructs an ExtractedFunc module.

    :param model: The nn.Module containing the layer.
    :param graph: The NNCF graph.
    :param input_node: The name of input node.
    :param output_node: The name of output node.
    :return: The extracted linear layer as an ExtractedFunc module.
    """
    if input_node != output_node:
        msg = "Only one input and output node supported."
        raise nncf.InternalError(msg)

    layer_attrs = input_node.layer_attributes

    if not isinstance(layer_attrs, PT2OpLayerAttributes):
        msg = f"Expected PT2OpLayerAttributes for input_node.layer_attributes, actual: {type(layer_attrs)}"
        raise nncf.InternalError(msg)

    weight_node = get_const_node(input_node, 1, graph)
    if weight_node is None:
        msg = f"Weight node not found for {input_node}"
        raise nncf.InternalError(msg)
    weight = get_const_data(weight_node, model)

    hook_storage = get_hook_storage(model)
    with torch.no_grad():
        # Calculate weight after execution all hook for weight data
        weight = hook_storage.execute_post_function_hooks(weight_node.node_name, 0, weight)
        weight = hook_storage.execute_pre_function_hooks(input_node.node_name, 1, weight)

    bias_node = get_const_node(input_node, 2, graph)
    bias = get_const_data(bias_node, model) if bias_node is not None else None

    layer_kwarg = {
        "weight": weight,
        "bias": bias,
    }
    linear_module = ExtractedFunc(layer_attrs.func, layer_kwarg)
    return linear_module


def extract_model(
    model: nn.Module, graph: PTNNCFGraph, input_nodes: list[str], output_nodes: list[str]
) -> Optional[nn.Module]:
    """
    Extracts a submodule from a given nn.Module containing only the nodes from the input to the output node.

    Supported subgraph:
      - Conv
      - Conv + BatchNorm
      - Linear

    :param model: The nn.Module to extract the submodule from.
    :param input_nodes: List containing names of the input nodes for the submodule.
    :param output_nodes: List containing names of the output nodes for the submodule.
    :return: An nn.Module containing the extracted submodel, or None if extraction is not supported.
    """
    if len(input_nodes) != 1 or len(output_nodes) != 1:
        msg = "input_nodes and output_nodes should contain only one node."
        raise nncf.InternalError(msg)

    input_node = graph.get_node_by_name(input_nodes[0])
    output_node = graph.get_node_by_name(output_nodes[0])

    if input_node.metatype in CONV_METATYPES:
        return extract_conv(model, graph, input_node, output_node)

    if input_node.metatype is om.PTLinearMetatype:
        return extract_linear(model, graph, input_node, output_node)

    nncf_logger.debug(f"Can not extract module for {input_node.node_name}")
    return None
