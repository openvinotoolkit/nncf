# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional, Tuple

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.layer_attributes import ConvolutionLayerAttributes
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.layer_attributes import OVLayerAttributes
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVDepthwiseConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSubtractMetatype
from nncf.openvino.graph.node_utils import create_bias_tensor
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import get_node_with_bias_value
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.openvino.statistics.collectors import OVQuantileReducer
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic
from nncf.quantization.algorithms.channel_alignment.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.channel_alignment.backend import ChannelAlignmentAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import LayoutDescriptor


@ALGO_BACKENDS.register(BackendType.OPENVINO)
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
        reduction_shape, q: float, num_samples: int, inplace: bool
    ) -> TensorStatisticCollectorBase:
        tensor_collector = TensorCollector(OVMinMaxTensorStatistic)
        quantile_reducer = OVQuantileReducer(reduction_shape, (q, 1 - q), inplace)

        for port_id, container_key in enumerate([OVMinMaxTensorStatistic.MIN_STAT, OVMinMaxTensorStatistic.MAX_STAT]):
            aggregator = MedianAggregator(OVNNCFCollectorTensorProcessor, num_samples=num_samples)
            tensor_collector.register_statistic_branch(container_key, quantile_reducer, aggregator, port_id)
        return tensor_collector

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        next_nodes = nncf_graph.get_next_nodes(node)
        if not next_nodes:
            return False

        add_node = next_nodes[0]
        if add_node.metatype != OVAddMetatype:
            return False

        bias_constant = get_node_with_bias_value(add_node, nncf_graph)
        return bias_constant is not None

    @staticmethod
    def create_bias_tensor(node: NNCFNode, nncf_graph: NNCFGraph, value: Any)-> np.ndarray:
        return create_bias_tensor(node, nncf_graph, value)
