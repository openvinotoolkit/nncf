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

from typing import Dict, List, Optional, Tuple

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVMultiplyInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.openvino.statistics.collectors import OVAbsMaxReducer
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend


class OVSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def convolution_metatype(self) -> OperatorMetatype:
        return OVConvolutionMetatype

    @property
    def matmul_metatype(self) -> OperatorMetatype:
        return OVMatMulMetatype

    @property
    def quantize_agnostic_metatypes(self) -> List[OperatorMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes and node.layer_attributes.constant_attributes

    @staticmethod
    def get_input_ports_map(node: NNCFNode, nncf_graph: NNCFGraph) -> Dict[str, int]:
        weight_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in weight_ports
        ]

        if len(weight_ports) != 1 or len(activation_ports) != 1:
            raise RuntimeError(f"Too many weight or activation ports for {node.node_name} node")

        return {"activation": activation_ports[0], "weight": weight_ports[0]}

    @staticmethod
    def get_channel_agnostic_reduction_axes(channel_axis: int, shape: Tuple[int]) -> Tuple[int]:
        return get_channel_agnostic_reduction_axes([channel_axis], shape)

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: Tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = OVAbsMaxReducer(reduction_axes=stats_reduction_axes, inplace=inplace)
        aggregator = MaxAggregator(tensor_processor=OVNNCFCollectorTensorProcessor, num_samples=num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def process_weight_statistics(weights: np.ndarray, reduction_shape: Tuple[int]) -> np.ndarray:
        return np.max(np.abs(weights), axis=reduction_shape)

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: ov.Model, port_id: int) -> np.ndarray:
        return get_weight_value(node_with_weight, model, port_id)

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        const_ids = node.layer_attributes.get_const_port_ids()
        if len(const_ids) != 1:
            raise RuntimeError(f"Found more than 1 port for {node.node_name} node")
        return const_ids[0]

    @staticmethod
    def clip_statistics(statistics: np.ndarray) -> np.ndarray:
        a_min = 1e-5
        squeezed = np.squeeze(statistics)
        return np.clip(squeezed, a_min=a_min, a_max=None)

    @staticmethod
    def calculate_scale_and_ratio(
        activations: np.ndarray, weights: np.ndarray, alpha: float, quantile: Optional[float] = 0.1
    ) -> np.ndarray:
        scales = np.power(activations, alpha) / (np.power(weights, 1 - alpha) + np.finfo(float).eps)

        a_min = np.quantile(scales, quantile)
        a_max = 1e2

        scales = np.clip(scales, a_min=a_min, a_max=a_max)
        ratio = scales.min() / (scales.max() + np.finfo(float).eps)
        return scales, ratio

    @staticmethod
    def calculate_activation_scale(scale_value: np.ndarray, activations_size: int, channel_axis: int) -> np.ndarray:
        activation_scale = scale_value ** (-1)
        if activations_size > 1:
            reshape_shape = np.ones(activations_size, dtype=np.int64)
            reshape_shape[channel_axis] = activation_scale.size
            activation_scale = np.reshape(activation_scale, reshape_shape)
        return activation_scale

    @staticmethod
    def calculate_weight_scale(scale_value: np.ndarray, weights_size: int, channel_axis: int) -> np.ndarray:
        weight_scale = scale_value
        if weights_size > 1:
            reshape_shape = np.ones(weights_size, dtype=np.int64)
            reshape_shape[channel_axis] = scale_value.size
            weight_scale = np.reshape(scale_value, reshape_shape)
        return weight_scale

    @staticmethod
    def weight_update_command(
        node_with_weight: NNCFNode, weight_value: np.ndarray, weight_port_id: int
    ) -> OVWeightUpdateCommand:
        return OVCommandCreator.create_command_to_update_weight(node_with_weight, weight_value, weight_port_id)

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode, scale_value: np.ndarray, port_id: int, nodes: List[NNCFNode], scale_node_name: str
    ) -> OVMultiplyInsertionCommand:
        return OVCommandCreator.multiply_insertion_command(source_node, nodes, port_id, scale_value, scale_node_name)

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        channel_axis = 1

        if port_id > 1:
            raise RuntimeError(f"{node.metatype.name} can not take more than 2 input tensors.")

        if node.metatype == OVMatMulMetatype:
            if (
                node.layer_attributes is not None
                and node.layer_attributes.input_attributes is not None
                and "transpose" in node.layer_attributes.input_attributes
            ):
                transpose = node.layer_attributes.input_attributes["transpose"]
                channel_axis = OVSmoothQuantAlgoBackend.calculate_port_based_channel_axis(port_id, transpose)

        return channel_axis

    @staticmethod
    def get_weight_channel_axis(node: NNCFNode, port_id: int) -> int:
        channel_axis = 1

        if port_id > 1:
            raise RuntimeError(f"{node.metatype.name} can not take more than 2 input tensors.")

        if port_id not in node.layer_attributes.constant_attributes:
            raise RuntimeError(f"{node.node_name} should contain {port_id} in the attributes map.")

        if node.metatype == OVMatMulMetatype:
            if "transpose" in node.layer_attributes.constant_attributes[port_id]:
                transpose = node.layer_attributes.constant_attributes[port_id]["transpose"]
                channel_axis = OVSmoothQuantAlgoBackend.calculate_port_based_channel_axis(port_id, transpose)

        return channel_axis

    @staticmethod
    def calculate_port_based_channel_axis(port_id: int, transpose: bool) -> int:
        return -2 + port_id if transpose else -1 - port_id
