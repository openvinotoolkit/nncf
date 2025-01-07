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

from typing import Any, Tuple

import numpy as np
import openvino.runtime as ov

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.openvino.graph import node_utils
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.layout import get_conv_weights_layout_from_node
from nncf.openvino.graph.layout import get_linear_weights_layout_from_node
from nncf.openvino.graph.metatypes.groups import CONV_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSubtractMetatype
from nncf.openvino.graph.node_utils import create_bias_tensor
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import OVQuantileReducer
from nncf.quantization.algorithms.channel_alignment.backend import ChannelAlignmentAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import LayoutDescriptor


class OVChannelAlignmentAlgoBackend(ChannelAlignmentAlgoBackend):
    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def get_bias_value(node: NNCFNode, model: ov.Model, nncf_graph: NNCFGraph) -> np.ndarray:
        return get_bias_value(node, nncf_graph, model)

    @staticmethod
    def get_weight_value(node: NNCFNode, model: ov.Model, port_id: int) -> np.ndarray:
        return get_weight_value(node, model, port_id)

    @staticmethod
    def get_activation_port_ids_for_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def get_weights_port_ids_for_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 1

    @staticmethod
    def get_conv_metatypes():
        return [OVConvolutionMetatype, OVGroupConvolutionMetatype, OVDepthwiseConvolutionMetatype]

    @staticmethod
    def get_linear_metatypes():
        return [OVMatMulMetatype]

    @staticmethod
    def get_add_metatypes():
        return [OVAddMetatype, OVSubtractMetatype]

    @staticmethod
    def get_statistic_collector(
        reduction_axes, q: float, num_samples: int, inplace: bool
    ) -> TensorStatisticCollectorBase:
        tensor_collector = TensorCollector(MinMaxTensorStatistic)
        quantile_reducer = OVQuantileReducer(reduction_axes, (q, 1 - q), inplace)

        for port_id, container_key in enumerate([MinMaxTensorStatistic.MIN_STAT, MinMaxTensorStatistic.MAX_STAT]):
            aggregator = MedianAggregator(num_samples=num_samples, aggregation_axes=(0, 1))
            tensor_collector.register_statistic_branch(container_key, quantile_reducer, aggregator, port_id)
        return tensor_collector

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return node_utils.is_node_with_bias(node, nncf_graph)

    @staticmethod
    def get_dims_descriptor(node: NNCFNode) -> LayoutDescriptor:
        if node.metatype in CONV_OPERATIONS:
            weights_layout = get_conv_weights_layout_from_node(node=node)
        elif node.metatype == OVMatMulMetatype:
            weights_layout = get_linear_weights_layout_from_node(node=node)
        else:
            raise nncf.InternalError(
                f"Metatype {node.metatype} of node {node.node_name} dimensions description retrieving is not supported"
            )

        if OVLayoutElem.GROUPS in weights_layout:
            # Using groups dim as output channels dim for ChannelAlignment algorithm
            # TODO(dlyakhov) support group convolutions with groups number not in [1, out_channels]
            return LayoutDescriptor(
                weights_layout.index(OVLayoutElem.GROUPS),
                weights_layout.index(OVLayoutElem.C_IN),
                node.metatype.output_channel_axis,
            )
        return LayoutDescriptor(
            weights_layout.index(OVLayoutElem.C_OUT) if OVLayoutElem.C_OUT in weights_layout else None,
            weights_layout.index(OVLayoutElem.C_IN),
            node.metatype.output_channel_axis,
        )

    @staticmethod
    def get_conv_layer_attributes(node: NNCFNode) -> ConvolutionLayerAttributes:
        return node.layer_attributes.layer_attributes

    @staticmethod
    def create_bias_tensor(node: NNCFNode, nncf_graph: NNCFGraph, value: Any) -> np.ndarray:
        return create_bias_tensor(node, nncf_graph, value)
