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

from typing import Dict, List, Tuple

import numpy as np

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNodeName
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.pruning.utils import get_output_channels
from nncf.common.pruning.utils import is_prunable_depthwise_conv


class WeightsFlopsCalculator:
    """
    Collection of weight and flops calculation functions.
    Class instance keeps only parameters that are constant during
    compression algorithms execution.
    """

    def __init__(self, conv_op_metatypes: List[OperatorMetatype], linear_op_metatypes: List[OperatorMetatype]):
        """
        Constructor.

        :param conv_op_metatypes: List of metatypes defining convolution operations.
        :param linear_op_metatypes: List of metatypes defining linear/fully connected operations.
        """
        self._conv_op_metatypes = conv_op_metatypes
        self._linear_op_metatypes = linear_op_metatypes

    def count_flops_and_weights(
        self,
        graph: NNCFGraph,
        output_shapes: Dict[NNCFNodeName, int],
        input_channels: Dict[NNCFNodeName, int] = None,
        output_channels: Dict[NNCFNodeName, int] = None,
        kernel_sizes: Dict[NNCFNodeName, Tuple[int, int]] = None,
        op_addresses_to_skip: List[str] = None,
    ) -> Tuple[int, int]:
        """
        Counts the number of weights and FLOPs in the model for convolution and fully connected layers.

        :param graph: NNCFGraph.
        :param output_shapes: Dictionary of output dimension shapes for convolutions and
            fully connected layers. E.g {node_name: (height, width)}
        :param input_channels: Dictionary of input channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param output_channels: Dictionary of output channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param kernel_sizes: Dictionary of kernel sizes in convolutions.
            If not specified, taken from the graph. {node_name: kernel_size}.
            It's only supposed to be used in NAS in case of Elastic Kernel enabled.
        :param op_addresses_to_skip: List of operation addresses of layers that should be skipped from calculation.
            It's only supposed to be used in NAS in case of Elastic Depth enabled.
        :return number of FLOPs for the model
                number of weights (params) in the model
        """
        flops_pers_node, weights_per_node = self.count_flops_and_weights_per_node(
            graph, output_shapes, input_channels, output_channels, kernel_sizes, op_addresses_to_skip
        )
        return sum(flops_pers_node.values()), sum(weights_per_node.values())

    def count_flops_and_weights_per_node(
        self,
        graph: NNCFGraph,
        output_shapes: Dict[NNCFNodeName, int],
        input_channels: Dict[NNCFNodeName, int] = None,
        output_channels: Dict[NNCFNodeName, int] = None,
        kernel_sizes: Dict[NNCFNodeName, Tuple[int, int]] = None,
        op_addresses_to_skip: List[NNCFNodeName] = None,
    ) -> Tuple[Dict[NNCFNodeName, int], Dict[NNCFNodeName, int]]:
        """
        Counts the number of weights and FLOPs per node in the model for convolution and fully connected layers.

        :param graph: NNCFGraph.
        :param output_shapes: Dictionary of output dimension shapes for convolutions and
            fully connected layers. E.g {node_name: (height, width)}
        :param input_channels: Dictionary of input channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param output_channels: Dictionary of output channels number in convolutions.
            If not specified, taken from the graph. {node_name: channels_num}
        :param kernel_sizes: Dictionary of kernel sizes in convolutions.
            If not specified, taken from the graph. {node_name: kernel_size}.
            It's only supposed to be used in NAS in case of Elastic Kernel enabled.
        :param op_addresses_to_skip: List of operation addresses of layers that should be skipped from calculation.
            It's only supposed to be used in NAS in case of Elastic Depth enabled.
        :return Dictionary of FLOPs number {node_name: flops_num}
                Dictionary of weights number {node_name: weights_num}
        """
        flops = {}
        weights = {}
        input_channels = input_channels or {}
        output_channels = output_channels or {}
        kernel_sizes = kernel_sizes or {}
        op_addresses_to_skip = op_addresses_to_skip or []
        for node in graph.get_nodes_by_metatypes(self._conv_op_metatypes):
            name = node.node_name
            if name in op_addresses_to_skip:
                continue
            num_in_channels = input_channels.get(name, node.layer_attributes.in_channels)
            num_out_channels = output_channels.get(name, node.layer_attributes.out_channels)
            kernel_size = kernel_sizes.get(name, node.layer_attributes.kernel_size)
            if is_prunable_depthwise_conv(node):
                # Prunable depthwise conv processed in special way
                # because common way to calculate filters per
                # channel for such layer leads to zero in case
                # some of the output channels are pruned.
                filters_per_channel = 1
            else:
                filters_per_channel = num_out_channels // node.layer_attributes.groups

            flops_numpy = (
                2 * np.prod(kernel_size) * num_in_channels * filters_per_channel * np.prod(output_shapes[name])
            )
            weights_numpy = np.prod(kernel_size) * num_in_channels * filters_per_channel

            flops[name] = flops_numpy.astype(int).item()
            weights[name] = weights_numpy.astype(int).item()

        for node in graph.get_nodes_by_metatypes(self._linear_op_metatypes):
            name = node.node_name
            if name in op_addresses_to_skip:
                continue

            num_in_features = input_channels.get(name, node.layer_attributes.in_features)
            num_out_features = output_channels.get(name, node.layer_attributes.out_features)

            flops_numpy = 2 * num_in_features * num_out_features * np.prod(output_shapes[name][:-1])
            weights_numpy = num_in_features * num_out_features
            flops[name] = flops_numpy
            weights[name] = weights_numpy

        return flops, weights

    def count_filters_num(self, graph: NNCFGraph, output_channels: Dict[NNCFNodeName, int] = None) -> int:
        """
        Counts filters of `op_metatypes` layers taking into account new output channels number.

        :param graph: NNCFGraph.
        :param output_channels:  A dictionary of output channels number in pruned model.
        :return: Current number of filters according to given graph and output channels.
        """
        filters_num = 0
        output_channels = output_channels or {}
        for node in graph.get_nodes_by_metatypes(self._conv_op_metatypes + self._linear_op_metatypes):
            filters_num += output_channels.get(node.node_name, get_output_channels(node))
        return filters_num
