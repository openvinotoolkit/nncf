from typing import Dict, List, Optional, Tuple

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import MedianAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.common import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVAddMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVGroupConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVSubtractMetatype
from nncf.openvino.graph.model_utils import insert_null_biases
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.openvino.statistics.collectors import OVQuantileReducer
from nncf.openvino.statistics.collectors import get_mean_batch_stat_collector
from nncf.openvino.statistics.statistics import OVMinMaxTensorStatistic
from nncf.openvino.tensor import OVNNCFTensor
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import ChannelAlignmentAlgoBackend
from nncf.quantization.algorithms.channel_alignment.backend import DimsDescriptor


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
    def get_conv_nodes(nncf_graph: NNCFGraph):
        return nncf_graph.get_nodes_by_metatypes([OVConvolutionMetatype, OVGroupConvolutionMetatype])

    @staticmethod
    def is_node_add_operation(node: NNCFNode) -> bool:
        return node.metatype in [OVAddMetatype, OVSubtractMetatype]

    @staticmethod
    def is_node_conv_or_matmul_operation(node: NNCFNode) -> bool:
        return node.metatype in [OVConvolutionMetatype, OVGroupConvolutionMetatype, OVMatMulMetatype]

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
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def create_bias_update_command(
        node_with_bias: NNCFNode, updated_value: np.ndarray, nncf_graph: NNCFGraph
    ) -> OVWeightUpdateCommand:
        return OVCommandCreator.create_command_to_update_bias(node_with_bias, updated_value, nncf_graph)

    @staticmethod
    def create_weights_update_command(
        node_with_weights: NNCFNode, updated_value: np.array, weights_port_id: int
    ) -> OVWeightUpdateCommand:
        return OVCommandCreator.create_command_to_update_weight(node_with_weights, updated_value, weights_port_id)

    @staticmethod
    def get_dims_descriptor(node: NNCFNode):
        return DimsDescriptor(
            conv_weight_out_channels_dim=0,
            conv_weight_in_channels_dim=1,
            bias_channels_dim=node.metatype.output_channel_axis,
        )

    @staticmethod
    def get_conv_layer_attributes(node: NNCFNode):
        return node.layer_attributes.common_layer_attrs[1]
