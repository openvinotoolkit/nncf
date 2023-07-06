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

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.openvino.graph.node_utils import get_weight_value
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVMultiplyInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.openvino.statistics.collectors import OVAbsMaxReducer
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.quantization.algorithms.smooth_quant.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def weighted_metatypes(self) -> List[OperatorMetatype]:
        return [OVMatMulMetatype]

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
            raise RuntimeError(f"Too many weights or activation ports for {node.node_name} node")

        return {"activation": activation_ports[0], "weight": weight_ports[0]}

    @staticmethod
    def calculate_input_reduction_shape(nncf_graph: NNCFGraph, node: NNCFNode, input_port: int) -> Tuple[int]:
        shape = nncf_graph.get_input_edges(node)[input_port].tensor_shape
        channels = shape[node.metatype.output_channel_axis]

        if node.layer_attributes.input_attributes["transpose"]:
            channels = shape[1]

        reduction_shape = tuple(i for i, val in enumerate(shape) if val != channels)
        return reduction_shape

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_shape: Tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = OVAbsMaxReducer(stats_reduction_shape, inplace)
        aggregator = MaxAggregator(OVNNCFCollectorTensorProcessor, num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_statistics(node: NNCFNode, model: ov.Model, port_id: int) -> np.ndarray:
        weights = deepcopy(get_weight_value(node, model, port_id))
        abs_value = np.abs(weights)
        transpose = node.layer_attributes.constant_attributes[port_id]["transpose"]
        axis = 0 if transpose else -1
        return np.max(abs_value, axis=axis)

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
    def calculate_activation_scale(scale_value: np.ndarray, nodes: List[NNCFNode]) -> np.ndarray:
        activation_scales = scale_value ** (-1)

        activation_shapes = [n.layer_attributes.input_attributes["shape"] for n in nodes]
        activation_shape = activation_shapes[0]
        if not all(shape == activation_shape for shape in activation_shapes):
            raise RuntimeError(f"Shapes for nodes {[n.node_name for n in nodes]} are not identical")

        transpose_attrs = [n.layer_attributes.input_attributes["transpose"] for n in nodes]
        if not all(attr == transpose_attrs[0] for attr in transpose_attrs):
            raise RuntimeError(f"Transpose attributes for nodes {[n.node_name for n in nodes]} are not identical")

        activation_scales = np.expand_dims(activation_scales, axis=0)

        if len(activation_shape) > 2:
            if all(transpose_attrs):
                activation_scales = np.expand_dims(activation_scales, axis=2)
            else:
                activation_scales = np.expand_dims(activation_scales, axis=1)
        return activation_scales

    @staticmethod
    def calculate_weight_scale(scale_value: np.ndarray, nodes: List[NNCFNode]) -> np.ndarray:
        transpose_attrs = []
        for node in nodes:
            port_id = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(node)
            transpose = node.layer_attributes.constant_attributes[port_id]["transpose"]
            transpose_attrs.append(transpose)

        if not all(attr == transpose_attrs[0] for attr in transpose_attrs):
            raise RuntimeError(f"Transpose attributes for nodes {[n.node_name for n in nodes]} are not identical")

        if all(transpose_attrs):
            weight_scales = np.expand_dims(scale_value, axis=0)
        else:
            weight_scales = np.expand_dims(scale_value, axis=-1)

        return weight_scales

    @staticmethod
    def weight_update_command(
        node_with_weight: NNCFNode, weight_value: np.ndarray, weight_port_id: int
    ) -> OVWeightUpdateCommand:
        return OVCommandCreator.create_command_to_update_weight(node_with_weight, weight_value, weight_port_id)

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode, scale_value: np.ndarray, port_id: int, nodes: List[NNCFNode]
    ) -> OVMultiplyInsertionCommand:
        return OVCommandCreator.multiply_insertion_command(source_node, nodes, port_id, scale_value)
