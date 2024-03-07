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

from copy import deepcopy
from itertools import chain
from typing import Iterable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from nncf import nncf_logger
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.torch.graph import operator_metatypes as om
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.model_graph_manager import get_fake_quantizer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer

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


class ExtractedConv(nn.Module):
    def __init__(
        self,
        fn_name: str,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        w_fq: Optional[BaseQuantizer],
        layer_attributes: ConvolutionLayerAttributes,
    ) -> None:
        super().__init__()
        assert fn_name in ["conv1d", "conv2d", "conv3d"]
        self.fn_name = fn_name
        self.layer_attributes = layer_attributes
        self.weight = weight.clone()
        self.w_fq = w_fq if w_fq is None else deepcopy(w_fq)
        self.bias = bias if bias is None else deepcopy(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_fq(self.weight) if self.w_fq is not None else self.weight
        return getattr(F, self.fn_name)(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.layer_attributes.stride,
            padding=self.layer_attributes.padding_values,
            dilation=self.layer_attributes.dilations,
            groups=self.layer_attributes.groups,
        )


class ExtractedConvTranspose(nn.Module):
    def __init__(
        self,
        fn_name: str,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        w_fq: Optional[BaseQuantizer],
        layer_attributes: ConvolutionLayerAttributes,
    ) -> None:
        super().__init__()
        self.fn_name = fn_name
        assert fn_name in ["conv_transpose1d", "conv_transpose2d", "conv_transpose3d"]
        self.layer_attributes = layer_attributes
        self.weight = weight.clone()
        self.w_fq = w_fq if w_fq is None else deepcopy(w_fq)
        self.bias = bias if bias is None else deepcopy(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.w_fq(self.weight) if self.w_fq is not None else self.weight
        return getattr(F, self.fn_name)(
            input=x,
            weight=w,
            bias=self.bias,
            stride=self.layer_attributes.stride,
            output_padding=self.layer_attributes.padding_values,
            dilation=self.layer_attributes.dilations,
        )


def extract_conv(node: NNCFNode, model: NNCFNetwork) -> ExtractedConv:
    """
    Extracts a convolutional layer from an NNCF graph and constructs an ExtractedConv module.
    :param node: The NNCF node representing the convolutional layer to extract.
    :param model: The NNCF network containing the layer.
    :return: The extracted convolutional layer as an ExtractedConv module.
    """
    graph = model.nncf.get_graph()
    weight_node = get_const_node(node, node.metatype.weight_port_ids[0], graph)
    weight = get_const_data(weight_node, model)
    w_fq = get_fake_quantizer(node, node.metatype.weight_port_ids[0], model)
    bias_node = get_const_node(node, node.metatype.bias_port_id, graph)
    bias = get_const_data(bias_node, model) if bias_node is not None else None
    if node.metatype in CONV_METATYPES:
        extracted_module = ExtractedConv(
            node.node_type, weight=weight, bias=bias, w_fq=w_fq, layer_attributes=node.layer_attributes
        )
    elif node.metatype in CONV_TRANSPOSE_METATYPES:
        extracted_module = ExtractedConvTranspose(
            node.node_type, weight=weight, bias=bias, w_fq=w_fq, layer_attributes=node.layer_attributes
        )
    return try_to_fuse_conv(node, model, extracted_module)


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
    If source modules inhered from nn.BatchNorm1d, nn.BatchNorm2d, or nn.BatchNorm3d return module that
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

    return extracted_bn


def try_to_fuse_conv(node: NNCFNode, model: NNCFNetwork, extracted_module: nn.Module):
    """
    Fused convolution operation with next batch if possible,

    :param node: _description_ #TODO
    :param model: _description_ #TODO
    :param target_module: _description_ #TODO
    """
    next_nodes = model.nncf.get_graph().get_next_nodes(node)
    if len(next_nodes) == 1:
        extracted_bn = extract_bn(next_nodes[0], model)
        if next_nodes[0].metatype == om.PTBatchNormMetatype:
            extracted_bn = extract_bn(next_nodes[0], model)
            if extracted_bn is None:
                nncf_logger.debug(
                    f"Can`t extract fused batchnorm module for {node.node_name},"
                    " module that contain batchnorm operator should be inhered from one of {BATCH_NORM_CLASSES}."
                )
                return None
            return nn.Sequential(extracted_module, extracted_bn)
    return extracted_module


def extract_fused_subgraph_for_node(node: NNCFNode, model: NNCFNetwork) -> Optional[nn.Sequential]:
    """
    Extract submodule with fused nodes to inference.
    If FQ exists for weights, FQ will be apply to weights.
    Supported submodules:
      - Conv
      - Conv+BN
    :param node: Target node.
    :param model: Source model.
    :return: nn.Sequential of fused subgraph for target node, overwise return None.
    """

    extracted_module: Optional[nn.Module] = None

    if node.metatype in CONV_METATYPES + CONV_TRANSPOSE_METATYPES:
        extracted_module = extract_conv(node, model)
    else:
        nncf_logger.debug(f"Can`t extract module for {node.node_name}")
        return None

    return extracted_module
