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
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.model_graph_manager import get_const_data
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
        return self.fn(input=x, **self.kwargs)


def apply_args_defaults(args: List[Any], kwargs: Dict[str, Any], indexed_args=List[Tuple[int, str]]) -> Dict[str, Any]:
    args_dict: Dict[str, Any] = dict()
    for idx, arg_name in enumerate(indexed_args):
        if idx < len(args):
            args_dict[arg_name] = args[idx]
        elif arg_name in kwargs:
            args_dict[arg_name] = kwargs[arg_name]

    return args_dict


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
    weight_node = get_const_node(input_node, input_node.metatype.weight_port_ids[0], graph)
    weight = get_const_data(weight_node, model)
    # w_fq = get_fake_quantizer(input_node, input_node.metatype.weight_port_ids[0], model)
    bias_node = get_const_node(input_node, input_node.metatype.bias_port_id, graph)
    bias = get_const_data(bias_node, model) if bias_node is not None else None

    # with torch.no_grad():
    #     e_weight = w_fq(weight) if w_fq else weight

    layer_attrs = input_node.layer_attributes

    if not isinstance(layer_attrs, PT2OpLayerAttributes):
        msg = "Expected PT2OpLayerAttributes for input_node.layer_attributes"
        raise nncf.InternalError(msg)

    conv_kwargs = apply_args_defaults(
        layer_attrs.op_args, layer_attrs.op_kwargs, [(3, "stride"), (4, "padding"), (5, "dilation"), (6, "groups")]
    )
    conv_kwargs["weight"] = weight
    conv_kwargs["bias"] = bias
    extracted_module = ExtractedFunc(input_node.node_type, conv_kwargs)

    # if input_node != output_node:
    #     extracted_module = try_to_fuse_conv(input_node, output_node, model, extracted_module)

    return extracted_module


def extract_model(
    model: nn.Module, graph: PTNNCFGraph, input_nodes: List[str], output_nodes: List[str]
) -> Optional[nn.Module]:
    """
    Extracts a submodule from a given NNCF network containing only the nodes from the input to the output node.

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
