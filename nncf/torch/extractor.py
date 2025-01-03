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

from copy import deepcopy
from itertools import chain
from typing import Any, Dict, Iterable, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

import nncf
from nncf import nncf_logger
from nncf.common.graph.graph import NNCFNode
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_fake_quantizer
from nncf.torch.nncf_network import NNCFNetwork

BATCH_NORM_CLASSES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


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
    def __init__(
        self,
        fn_name: str,
        kwargs: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.fn_name = fn_name
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return getattr(F, self.fn_name)(input=x, **self.kwargs)


def extract_conv(
    input_node: NNCFNode,
    output_node: NNCFNode,
    model: NNCFNetwork,
) -> ExtractedFunc:
    """
    Extracts a convolutional layer from an NNCF graph and constructs an ExtractedFunc module.

    :param input_nodes: The name of input node.
    :param output_nodes: The name of output node.
    :param model: The NNCF network containing the layer.
    :return: The extracted convolutional layer as an ExtractedFunc module.
    """
    graph = model.nncf.get_graph()
    weight_node = get_const_node(input_node, input_node.metatype.weight_port_ids[0], graph)
    weight = get_const_data(weight_node, model)
    w_fq = get_fake_quantizer(input_node, input_node.metatype.weight_port_ids[0], model)
    bias_node = get_const_node(input_node, input_node.metatype.bias_port_id, graph)
    bias = get_const_data(bias_node, model) if bias_node is not None else None

    with torch.no_grad():
        e_weight = w_fq(weight) if w_fq else weight

    if input_node.metatype in CONV_METATYPES:
        kwargs = {
            "weight": e_weight.clone(),
            "bias": bias.clone() if bias is not None else bias,
            "stride": input_node.layer_attributes.stride,
            "padding": input_node.layer_attributes.padding_values,
            "dilation": input_node.layer_attributes.dilations,
            "groups": input_node.layer_attributes.groups,
        }
    elif input_node.metatype in CONV_TRANSPOSE_METATYPES:
        kwargs = {
            "weight": e_weight.clone(),
            "bias": bias.clone() if bias is not None else bias,
            "stride": input_node.layer_attributes.stride,
            "padding": input_node.layer_attributes.padding_values,
            "output_padding": input_node.layer_attributes.output_padding_values,
            "dilation": input_node.layer_attributes.dilations,
        }

    extracted_module = ExtractedFunc(input_node.node_type, kwargs)

    if input_node != output_node:
        extracted_module = try_to_fuse_conv(input_node, output_node, model, extracted_module)

    return extracted_module


def _find_parent_class(cls: type, parent_classes: Iterable[type]) -> Optional[type]:
    """
    Finds the first parent class of the given class that is present in the list of possible parent classes.

    :param cls: The class whose parent to find.
    :param parent_classes: A list of potential parent classes.
    :return: The first matching parent class, or None if no match is found.
    """
    for exp_cls in parent_classes:
        if issubclass(cls, exp_cls):
            return exp_cls
    return None


def extract_bn(node: NNCFNode, model: NNCFNetwork) -> Optional[Union[nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]]:
    """
    Extract batch_norm operation.
    If source modules inhered from nn.BatchNorm1d, nn.BatchNorm2d, or nn.BatchNorm3d return torch BatchNorm module.

    :param node: Target batch_norm node.
    :param model: Source model.
    :return: BatchNorm module with same attributes and parameters from source module or None.
    """
    bn_module: _BatchNorm = model.nncf.get_containing_module(node.node_name)
    bn_class = _find_parent_class(bn_module.__class__, BATCH_NORM_CLASSES)
    if bn_class is None:
        nncf_logger.debug(f"Module associated with {node} should be inhered from one of {BATCH_NORM_CLASSES}")
        return None

    extracted_bn: _BatchNorm = bn_class(
        num_features=bn_module.num_features,
        eps=bn_module.eps,
        momentum=bn_module.momentum,
        affine=bn_module.affine,
        track_running_stats=bn_module.track_running_stats,
        device=bn_module.weight.device,
        dtype=bn_module.weight.dtype,
    )

    # Copy named parameters and buffer that exists in native BatchNorm module from module in the module.
    for name, _ in chain(extracted_bn.named_parameters(), extracted_bn.named_buffers()):
        setattr(extracted_bn, name, deepcopy(getattr(bn_module, name)))
    extracted_bn.eval()
    return extracted_bn


def try_to_fuse_conv(
    input_node: NNCFNode, output_node: NNCFNode, model: NNCFNetwork, extracted_module: nn.Module
) -> nn.Module:
    """
    Fused convolution operation with the next batch norm node if possible,

    :param input_node: Input subgraph node.
    :param output_node: Output subgraph node (fused with input node).
    :param model: Source model.
    :param extracted_module: Extracted module.
    """
    next_nodes = model.nncf.get_graph().get_next_nodes(input_node)

    if len(next_nodes) != 1:
        return extracted_module

    if output_node != next_nodes[0]:
        raise nncf.InternalError(f"Output node {output_node} not found after {input_node}")

    if next_nodes[0].metatype != om.PTBatchNormMetatype:
        raise nncf.InternalError("Supported only BatchNorm layers")

    extracted_bn = extract_bn(next_nodes[0], model)
    if extracted_bn is None:
        nncf_logger.debug(
            f"Can`t extract fused batchnorm module for {input_node.node_name},"
            " module that contain batchnorm operator should be inhered from one of {BATCH_NORM_CLASSES}."
        )
        return None
    return nn.Sequential(extracted_module, extracted_bn)


def extract_model(model: NNCFNetwork, input_nodes: List[str], output_nodes: List[str]) -> Optional[nn.Module]:
    """
    Extracts a submodule from a given NNCF network containing only the nodes from the input to the output node.

    :param model: The NNCF network to extract the submodule from.
    :param input_nodes: List containing names of the input nodes for the submodule.
    :param output_nodes: List containing names of the output nodes for the submodule.
    :return: An nn.Module containing the extracted submodel, or None if extraction is not supported.
    """

    if len(input_nodes) != 1 or len(output_nodes) != 1:
        raise nncf.InternalError("input_nodes and output_nodes should contain only one node.")

    graph = model.nncf.get_graph()
    input_node = graph.get_node_by_name(input_nodes[0])
    output_node = graph.get_node_by_name(output_nodes[0])

    if input_node.metatype in CONV_METATYPES + CONV_TRANSPOSE_METATYPES:
        return extract_conv(input_node, output_node, model)

    nncf_logger.debug(f"Can`t extract module for {input_node.node_name}")
    return None
