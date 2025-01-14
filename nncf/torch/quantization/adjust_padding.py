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
from collections import namedtuple
from typing import NamedTuple

import networkx as nx
import torch

import nncf
from nncf.common.graph import NNCFNodeName
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.layers import NNCFConv2d
from nncf.torch.module_operations import UpdatePaddingValue
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import QuantizerConfig
from nncf.torch.quantization.layers import SymmetricQuantizer


class AdjustPaddingArgs(NamedTuple):
    weight_bitwidth: int
    activation_quantizer: BaseQuantizer
    module_op_node_name: NNCFNodeName


class CalculatePaddingAdjustment:
    """
    Calculates padding value to perform a workaround for U4 support on NPU.
    NPU supports only i4 for weights and activations with zero-point=0 and padding=0. This imposes some limitations on
    the quantization scheme we can apply. In case of unsigned input for a quantizer (e.g. output of ReLU) half of
    i4 range (8 values) is insufficient to preserve the accuracy. To overcome the problem it is proposed
    to transform u4 to i4 in the NPU plugin by shifting the input by half of the quantization range to left. Padding
    value should be shifted as well. And to make it zero after the shift (non-zero padding values are not
    supported), the model should be trained with padding value equal to the half of the quantization range.
    """

    def __init__(self, activation_quantizer: SymmetricQuantizer):
        if not isinstance(activation_quantizer, SymmetricQuantizer):
            raise nncf.InternalError("Padding adjustment is not supported for not symmetric quantization")
        self._activation_quantizer = activation_quantizer
        self._is_enabled = True

    def __call__(self, previous_padding_value) -> torch.Tensor:
        if self._is_enabled:
            scale = self._activation_quantizer.scale
            eps = self._activation_quantizer.eps
            safe_scale = abs(scale) + eps
            return safe_scale / 2
        return previous_padding_value

    @staticmethod
    def is_config_applicable(qconfig: QuantizerConfig):
        return (
            not qconfig.per_channel
            and qconfig.num_bits == 4
            and not qconfig.signedness_to_force
            and qconfig.mode == QuantizationMode.SYMMETRIC
        )


def add_adjust_padding_nodes(bitwidth_graph: nx.DiGraph, model: NNCFNetwork) -> nx.DiGraph():
    NewNodeArgs = namedtuple("NewNodeArgs", ("node_key", "attr", "parent_node_key"))
    nncf_graph = model.nncf.get_graph()
    args = []
    for node_key in bitwidth_graph.nodes:
        node = nncf_graph.get_node_by_key(node_key)
        module = model.nncf.get_containing_module(node.node_name)
        if isinstance(module, NNCFConv2d):
            adjust_padding_ops = filter(lambda x: isinstance(x, UpdatePaddingValue), module.pre_ops.values())
            for _ in adjust_padding_ops:
                new_node_key = f"{node_key}_apad"
                attr = dict(type="", label="adjust_padding_value", style="filled", color="yellow")
                args.append(NewNodeArgs(new_node_key, attr, node_key))

    for arg in args:
        bitwidth_graph.add_node(arg.node_key, **arg.attr)
        bitwidth_graph.add_edge(arg.node_key, arg.parent_node_key)
    return bitwidth_graph
