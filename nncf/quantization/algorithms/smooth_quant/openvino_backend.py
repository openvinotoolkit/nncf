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
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_shape
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
        reduction_shape = tuple([0])
        if len(shape) > 1:
            channel_axis = OVSmoothQuantAlgoBackend._get_activation_channel_axis(node, input_port)
            reduction_shape = get_channel_agnostic_reduction_shape([channel_axis], shape)
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
        reduction_axis = 0
        if len(abs_value.shape) > 1:
            reduction_axis = OVSmoothQuantAlgoBackend._get_weight_reduction_axis(node, port_id)
        return np.max(abs_value, axis=reduction_axis)

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
    def calculate_activation_scale(scale_value: np.ndarray, nodes: List[NNCFNode], nncf_graph: NNCFGraph) -> np.ndarray:
        activation_scale = scale_value ** (-1)

        activation_shapes = [n.layer_attributes.input_attributes["shape"] for n in nodes]
        activation_shape = activation_shapes[0]
        if not all(shape == activation_shape for shape in activation_shapes):
            raise RuntimeError(f"Shapes for nodes {[n.node_name for n in nodes]} are not identical")

        activation_ports_map = {
            node: OVSmoothQuantAlgoBackend.get_input_ports_map(node, nncf_graph)["activation"] for node in nodes
        }
        channel_axes = [
            OVSmoothQuantAlgoBackend._get_activation_channel_axis(node, port)
            for node, port in activation_ports_map.items()
        ]
        channel_axis = channel_axes[0]

        if not all(axis == channel_axis for axis in channel_axes):
            raise RuntimeError(f"Channel axes for nodes {[n.node_name for n in nodes]} are not identical")

        if len(activation_shape) > 1:
            reshape_shape = np.ones(len(activation_shape), dtype=np.int64)
            reshape_shape[channel_axis] = activation_shape[channel_axis]
            activation_scale = np.reshape(activation_scale, reshape_shape)

        return activation_scale

    @staticmethod
    def calculate_weight_scale(scale_value: np.ndarray, node: NNCFNode) -> np.ndarray:
        port_id = OVSmoothQuantAlgoBackend.get_weight_tensor_port_id(node)
        shape = node.layer_attributes.constant_attributes[port_id]["shape"]
        if len(shape) > 1:
            reduction_axis = OVSmoothQuantAlgoBackend._get_weight_reduction_axis(node, port_id)
            return np.expand_dims(scale_value, axis=reduction_axis)
        return scale_value

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

    @staticmethod
    def _get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        """
        Returns axis number of the activation tensor which correspond to it channel.

        :param node: NNCFNode instance.
        :param port_id: Specified input port id.
        :return: Channel axis number.
        """
        channel_axis = 1

        if node.metatype == OVMatMulMetatype:
            if port_id > 1:
                raise RuntimeError(f"{OVMatMulMetatype.name} can not take more than 2 input tensors.")

            channel_axis = -1 - port_id
            if (
                node.layer_attributes is not None
                and node.layer_attributes.input_attributes is not None
                and "transpose" in node.layer_attributes.input_attributes
                and node.layer_attributes.input_attributes["transpose"]
            ):
                channel_axis = -2 + port_id

        return channel_axis

    @staticmethod
    def _get_weight_reduction_axis(node: NNCFNode, port_id: int) -> int:
        """
        Returns axis number of the weight tensor which correspond to it channel.

        :param node: NNCFNode instance.
        :param port_id: Specified input port id.
        :return: Channel axis number.
        """
        channel_axis = 1 if node.metatype.const_channel_axis is None else node.metatype.const_channel_axis[0]

        if node.metatype == OVMatMulMetatype:
            if port_id > 1:
                raise RuntimeError(f"{OVMatMulMetatype.name} can not take more than 2 input tensors.")

            channel_axis = -2 + port_id
            if (
                node.layer_attributes is not None
                and node.layer_attributes.constant_attributes is not None
                and "transpose" in node.layer_attributes.constant_attributes[port_id]
                and node.layer_attributes.constant_attributes[port_id]["transpose"]
            ):
                channel_axis = -1 - port_id

        return channel_axis
