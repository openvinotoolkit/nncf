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
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn

import nncf
from nncf import nncf_logger
from nncf.common.graph.graph import NNCFNode
from nncf.experimental.torch2.function_hook.nncf_graph.layer_attributes import PT2OpLayerAttributes
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.model_graph_manager import get_const_data, get_input_fake_quantize_node
from nncf.torch.model_graph_manager import get_const_node

CONV_METATYPES = (
    om.PTConv1dMetatype,
    om.PTConv2dMetatype,
    om.PTConv3dMetatype,
    om.PTDepthwiseConv1dSubtype,
    om.PTDepthwiseConv2dSubtype,
    om.PTDepthwiseConv3dSubtype,
)

CONV_TRANSPOSE_METATYPES = (
    om.PTConvTranspose1dMetatype,
    om.PTConvTranspose2dMetatype,
    om.PTConvTranspose3dMetatype,
)


class ExtractedFunc(nn.Module):
    def __init__(self, fn: Callable[..., Any], kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.fn = fn
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x, **self.kwargs)


def apply_args_defaults(args: List[Any], kwargs: Dict[str, Any], indexed_args=List[Tuple[int, str]]) -> Dict[str, Any]:
    args_dict: Dict[str, Any] = dict()
    for idx, arg_name in indexed_args:
        if idx < len(args):
            args_dict[arg_name] = args[idx]
        elif arg_name in kwargs:
            args_dict[arg_name] = kwargs[arg_name]

    return args_dict


def extract_bn(model: nn.Module, graph: PTNNCFGraph, node: NNCFNode) -> ExtractedFunc:
    """
    Extract batch_norm operation.
    If source modules inhered from nn.BatchNorm1d, nn.BatchNorm2d, or nn.BatchNorm3d return torch BatchNorm module.

    :param node: Target batch_norm node.
    :param model: Source model.
    :return: BatchNorm module with same attributes and parameters from source module or None.
    """
    layer_attr = node.layer_attributes
    if not isinstance(layer_attr, PT2OpLayerAttributes):
        msg = f"Expected PT2OpLayerAttributes for input_node.layer_attributes, actual: {type(layer_attr)}"
        raise nncf.InternalError(msg)
    
    weigth_node = get_const_node(node, 1, graph)
    weight = get_const_data(weigth_node, model)

    bias_node = get_const_node(node, 2, graph)
    bias = get_const_data(bias_node, model)

    running_mean_node = get_const_node(node, 3, graph)
    running_mean = get_const_data(running_mean_node, model)

    running_var_node = get_const_node(node, 4, graph)
    running_var = get_const_data(running_var_node, model)

    bn_kwargs = apply_args_defaults(
        layer_attr.op_args, layer_attr.op_kwargs, [(6, "momentum"), (7, "eps"), (8, "cudnn_enabled")]
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
) -> ExtractedFunc:
    """
    Extracts a convolutional layer from an NNCF graph and constructs an ExtractedFunc module.

    :param model: The NNCF network containing the layer.
    :param graph: The NNCF graph.
    :param input_nodes: The name of input node.
    :param output_nodes: The name of output node.
    :return: The extracted convolutional layer as an ExtractedFunc module.
    """
    weight_node = get_const_node(input_node, 1, graph)
    weight = get_const_data(weight_node, model)

    hook_storage = get_hook_storage(model)
    with torch.no_grad():
        # Calculate weight after executuin all hook fro weight data
        weight = hook_storage.execute_post_function_hooks(weight_node.node_name, 0, weight)
        weight = hook_storage.execute_pre_function_hooks(input_node.node_name, 1, weight)

    bias_node = get_const_node(input_node, 2, graph)
    bias = get_const_data(bias_node, model) if bias_node is not None else None

    layer_attrs = input_node.layer_attributes

    if not isinstance(layer_attrs, PT2OpLayerAttributes):
        msg = f"Expected PT2OpLayerAttributes for input_node.layer_attributes, actual: {type(layer_attrs)}"
        raise nncf.InternalError(msg)

    conv_kwargs = apply_args_defaults(
        layer_attrs.op_args, layer_attrs.op_kwargs, [(3, "stride"), (4, "padding"), (5, "dilation"), (6, "groups")]
    )
    conv_kwargs["weight"] = weight
    conv_kwargs["bias"] = bias
    conv_module = ExtractedFunc(layer_attrs.func, conv_kwargs)

    if input_node == output_node:
        return conv_module
    
    if not output_node.metatype is om.PTBatchNormMetatype:
        msg = f"Support only PTBatchNormMetatype as output node, actual: {output_node.metatype}"
        raise nncf.InternalError(msg)
    
    next_nodes = graph.get_next_nodes(input_node)
    if output_node not in next_nodes:
        msg = f"Output node {output_node} not found after {input_node}"
        raise nncf.InternalError(msg)
    
    bn_module = extract_bn(model, graph, output_node)
    return nn.Sequential(conv_module, bn_module)


def extract_model(
    model: nn.Module, graph: PTNNCFGraph, input_nodes: List[str], output_nodes: List[str]
) -> Optional[nn.Module]:
    """
    Extracts a submodule from a given NNCF network containing only the nodes from the input to the output node.

    Supported subgraph:
      - Conv
      - Conv + BatchNorm
    
    :param model: The NNCF network to extract the submodule from.
    :param input_nodes: List containing names of the input nodes for the submodule.
    :param output_nodes: List containing names of the output nodes for the submodule.
    :return: An nn.Module containing the extracted submodel, or None if extraction is not supported.
    """

    if len(input_nodes) != 1 or len(output_nodes) != 1:
        raise nncf.InternalError("input_nodes and output_nodes should contain only one node.")

    input_node = graph.get_node_by_name(input_nodes[0])
    output_node = graph.get_node_by_name(output_nodes[0])

    if input_node.metatype in CONV_METATYPES:
        return extract_conv(model, graph, input_node, output_node)

    nncf_logger.debug(f"Can`t extract module for {input_node.node_name}")
    return None
