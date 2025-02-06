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

from typing import Callable, List, Tuple

import numpy as np
import openvino.runtime as ov

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.layout import OVLayoutElem
from nncf.openvino.graph.layout import get_linear_weights_layout_from_node
from nncf.openvino.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVMultiplyInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.openvino.statistics.collectors import OVAbsMaxReducer
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from nncf.tensor import Tensor

OV_PRE_LAYER_TARGET_TYPE = TargetType.PRE_LAYER_OPERATION


class OVSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [OVConvolutionMetatype]

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [OVMatMulMetatype]

    @property
    def quantize_agnostic_metatypes(self) -> List[OperatorMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS

    @staticmethod
    def pre_layer_target_type() -> TargetType:
        return OV_PRE_LAYER_TARGET_TYPE

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def get_activations_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        weight_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in weight_ports
        ]

        if len(activation_ports) != 1:
            raise nncf.InternalError(f"Too many weight or activation ports for {node.node_name} node")
        return activation_ports[0]

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: Tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = OVAbsMaxReducer(reduction_axes=stats_reduction_axes, inplace=inplace)
        aggregator = MaxAggregator(num_samples=num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: ov.Model, nncf_graph: NNCFGraph) -> Tensor:
        port_id = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(node_with_weight)
        return Tensor(get_weight_value(node_with_weight, model, port_id))

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        const_ids = node.layer_attributes.get_const_port_ids()
        if len(const_ids) != 1:
            raise nncf.InternalError(f"Found more than 1 port for {node.node_name} node")
        return const_ids[0]

    @staticmethod
    def weight_update_command(node_with_weight: NNCFNode, weight_value: np.ndarray) -> OVWeightUpdateCommand:
        weight_port_id = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(node_with_weight)
        return OVCommandCreator.create_command_to_update_weight(node_with_weight, weight_value, weight_port_id)

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode,
        scale_value: np.ndarray,
        source_output_port_id: int,
        nodes: List[NNCFNode],
        scale_node_name: str,
    ) -> OVMultiplyInsertionCommand:
        return OVCommandCreator.multiply_insertion_command(
            source_node, nodes, source_output_port_id, scale_value, scale_node_name
        )

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        channel_axis = 1

        if port_id > 1:
            raise nncf.InternalError(f"{node.metatype.name} can not take more than 2 input tensors.")

        if (
            node.metatype == OVMatMulMetatype
            and node.layer_attributes is not None
            and node.layer_attributes.input_attributes is not None
            and "transpose" in node.layer_attributes.input_attributes
        ):
            transpose = node.layer_attributes.input_attributes["transpose"]
            channel_axis = OVSmoothQuantAlgoBackend.calculate_port_based_channel_axis(port_id, transpose)

        return channel_axis

    @staticmethod
    def get_weight_channel_axis(node: NNCFNode) -> int:
        if node.metatype != OVMatMulMetatype:
            return 1

        weights_layout = get_linear_weights_layout_from_node(node)
        return weights_layout.index(OVLayoutElem.C_IN)

    @staticmethod
    def calculate_port_based_channel_axis(port_id: int, transpose: bool) -> int:
        return -2 + port_id if transpose else -1 - port_id

    @staticmethod
    def is_node_with_shared_weight(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        weight_port_id = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(node)
        weight_node = nncf_graph.get_input_edge_by_port_id(node, weight_port_id).from_node
        return len(nncf_graph.get_next_nodes(weight_node)) > 1

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == OV_PRE_LAYER_TARGET_TYPE
                and point.target_point.port_id == activation_port_id
            )

        return filter_func
