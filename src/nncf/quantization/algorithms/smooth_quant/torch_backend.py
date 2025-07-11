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

import torch

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from nncf.tensor import Tensor
from nncf.torch.function_hook.commands import PT2ConstUpdateCommand
from nncf.torch.function_hook.commands import PT2InsertionCommand
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTWeightUpdateCommand
from nncf.torch.model_graph_manager import get_const_data
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.layers import SQMultiply

PT_PRE_LAYER_TARGET_TYPE = TargetType.OPERATOR_PRE_HOOK


class PTSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def convolution_metatypes(self) -> list[OperatorMetatype]:
        return [
            om.PTConv1dMetatype,
            om.PTConv2dMetatype,
            om.PTConv3dMetatype,
        ]

    @property
    def matmul_metatypes(self) -> list[OperatorMetatype]:
        return [om.PTLinearMetatype]

    @property
    def quantize_agnostic_metatypes(self) -> list[OperatorMetatype]:
        return DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.QUANTIZATION_AGNOSTIC]

    @staticmethod
    def pre_layer_target_type() -> TargetType:
        return PT_PRE_LAYER_TARGET_TYPE

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        # Metatypes of linear and convolution operators guarantee
        # all nodes with the metatypes have weights, we can skip
        # this check by returning True.
        return True

    @staticmethod
    def get_activations_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        # Metatypes of linear and convolution operators guarantee
        # all nodes with the metatypes have 0 activation port id.
        return 0

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = AbsMaxReducer(reduction_axes=stats_reduction_axes)
        aggregator = MaxAggregator(num_samples=num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: NNCFNetwork, nncf_graph: NNCFGraph) -> Tensor:
        if isinstance(model, GraphModelWrapper):
            model = model.model

        weight_node = get_const_node(node_with_weight, node_with_weight.metatype.weight_port_ids[0], nncf_graph)
        if weight_node is None:
            msg = f"{node_with_weight} node has no weight node."
            raise RuntimeError(msg)
        weight_data = get_const_data(weight_node, model)
        return Tensor(weight_data)

    @staticmethod
    def weight_update_command(
        node_with_weight: NNCFNode, nncf_graph: NNCFGraph, weight_value: torch.Tensor
    ) -> PTWeightUpdateCommand:
        weight_node = get_const_node(node_with_weight, node_with_weight.metatype.weight_port_ids[0], nncf_graph)
        return PT2ConstUpdateCommand(weight_node, weight_value)

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode,
        scale_value: torch.Tensor,
        source_output_port_id: int,
        nodes: list[NNCFNode],
        scale_node_name: str,
    ) -> PTSharedFnInsertionCommand:
        input_port_id = 0
        target_points = []
        for node in nodes:
            target_points.append(PTTargetPoint(PT_PRE_LAYER_TARGET_TYPE, node.node_name, input_port_id=input_port_id))

        sq_multiply = SQMultiply(scale_value.shape)
        sq_multiply.scale = scale_value

        return PT2InsertionCommand(target_points=target_points, hook_module=sq_multiply)

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        if node.metatype == om.PTLinearMetatype:
            return -1
        # TODO: Add activation axis calculation when MatMul will be supported
        return 1

    @staticmethod
    def get_weight_channel_axis(node: NNCFNode) -> int:
        # TODO: Add activation axis calculation when MatMul will be supported
        return 1

    @staticmethod
    def is_node_with_shared_weight(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        weight_node = get_const_node(node, node.metatype.weight_port_ids[0], nncf_graph)
        output_edges = nncf_graph.get_next_nodes(weight_node)
        return len(output_edges) > 1

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == PT_PRE_LAYER_TARGET_TYPE
                and point.target_point.input_port_id == activation_port_id
            )

        return filter_func
