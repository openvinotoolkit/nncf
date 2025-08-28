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

from typing import Callable

import numpy as np
import onnx

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.onnx.graph.metatypes.groups import MATMUL_METATYPES
from nncf.onnx.graph.metatypes.groups import OPERATIONS_WITH_WEIGHTS
from nncf.onnx.graph.metatypes.groups import QUANTIZE_AGNOSTIC_OPERATIONS
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.node_utils import get_act_quantization_axis
from nncf.onnx.graph.node_utils import get_weight_quantization_axis
from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.graph.transformations.command_creation import ONNXCommandCreator
from nncf.onnx.graph.transformations.commands import ONNXInitializerUpdateCommand
from nncf.onnx.graph.transformations.commands import ONNXMultiplyInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from nncf.tensor import Tensor


class ONNXSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        return [ONNXConvolutionMetatype]

    @property
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        return MATMUL_METATYPES

    @property
    def quantize_agnostic_metatypes(self) -> list[OperatorMetatype]:
        return QUANTIZE_AGNOSTIC_OPERATIONS

    @staticmethod
    def pre_layer_target_type() -> TargetType:
        return TargetType.PRE_LAYER_OPERATION

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.metatype in OPERATIONS_WITH_WEIGHTS and node.layer_attributes.has_weight()

    @staticmethod
    def get_activations_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        weight_ports = list(node.layer_attributes.weight_attrs.keys())
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in weight_ports
        ]

        if len(activation_ports) != 1:
            msg = f"Too many weight or activation ports for {node.node_name} node"
            raise nncf.InternalError(msg)
        return activation_ports[0]

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = AbsMaxReducer(reduction_axes=stats_reduction_axes, inplace=inplace)
        aggregator = MaxAggregator(num_samples=num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def _get_weight_tensor_port_id(node: NNCFNode) -> int:
        weight_ports = list(node.layer_attributes.weight_attrs.keys())
        if len(weight_ports) != 1:
            msg = f"Found more than 1 port for {node.node_name} node"
            raise nncf.InternalError(msg)
        return weight_ports[0]

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: onnx.ModelProto, nncf_graph: NNCFGraph) -> Tensor:
        port_id = ONNXSmoothQuantAlgoBackend._get_weight_tensor_port_id(node_with_weight)
        weight_name = node_with_weight.layer_attributes.weight_attrs[port_id]["name"]
        return Tensor(get_tensor_value(model, weight_name))

    @staticmethod
    def weight_update_command(
        node_with_weight: NNCFNode, nncf_graph: NNCFGraph, weight_value: np.ndarray
    ) -> ONNXInitializerUpdateCommand:
        weight_port_id = ONNXSmoothQuantAlgoBackend._get_weight_tensor_port_id(node_with_weight)
        return ONNXCommandCreator.create_command_to_update_weight(node_with_weight, weight_value, weight_port_id)

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode,
        scale_value: np.ndarray,
        source_output_port_id: int,
        nodes: list[NNCFNode],
        scale_node_name: str,
    ) -> ONNXMultiplyInsertionCommand:
        return ONNXCommandCreator.multiply_insertion_command(
            source_node, nodes, source_output_port_id, scale_value, scale_node_name
        )

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        return get_act_quantization_axis(node, port_id)

    @staticmethod
    def get_weight_channel_axis(node: NNCFNode) -> int:
        port_id = ONNXSmoothQuantAlgoBackend._get_weight_tensor_port_id(node)
        return get_weight_quantization_axis(node, port_id)

    @staticmethod
    def is_node_with_shared_weight(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return node.is_shared

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == TargetType.PRE_LAYER_OPERATION
                and point.target_point.port_id == activation_port_id
            )

        return filter_func
