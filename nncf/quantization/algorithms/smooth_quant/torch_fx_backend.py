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

import torch.fx

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
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.node_utils import get_tensor_constant_from_node
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder
from nncf.experimental.torch.fx.transformations import get_graph_node_by_name
from nncf.experimental.torch.fx.transformations import module_insertion_transformation_builder
from nncf.openvino.graph.transformations.commands import OVMultiplyInsertionCommand
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from nncf.tensor import Tensor
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import get_const_node
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT

PT_PRE_LAYER_TARGET_TYPE = TargetType.OPERATOR_PRE_HOOK


class FXSQMultiply(torch.nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self._scale_value = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mul(x, self._scale_value)


class FXSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.PTConv1dMetatype,
            om.PTConv2dMetatype,
            om.PTConv3dMetatype,
        ]

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTLinearMetatype]

    @property
    def quantize_agnostic_metatypes(self) -> List[OperatorMetatype]:
        return DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.QUANTIZATION_AGNOSTIC]

    @staticmethod
    def pre_layer_target_type() -> TargetType:
        return PT_PRE_LAYER_TARGET_TYPE

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        # Metatypes of linears and convolutions guarantee
        # all nodes with the metatypes have weights, we can skip
        # this check by returning True.
        return True

    @staticmethod
    def get_activations_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        return 0

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: Tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = AbsMaxReducer(reduction_axes=stats_reduction_axes)
        aggregator = MaxAggregator(num_samples=num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> Tensor:
        weight_node = get_const_node(node_with_weight, node_with_weight.metatype.weight_port_ids[0], nncf_graph)
        if weight_node is None:
            raise RuntimeError(f"{node_with_weight} node has no weight node.")
        graph_node = get_graph_node_by_name(model.graph, weight_node.node_name)
        weight_data = get_tensor_constant_from_node(graph_node, model)
        return Tensor(weight_data.data)

    @staticmethod
    def weight_update_command(node_with_weight: NNCFNode, weight_value: torch.Tensor) -> OVWeightUpdateCommand:
        # TODO(dlyakhov): Use input port id depending on the node metatype/attributes.
        return FXApplyTransformationCommand(
            constant_update_transformation_builder(node_with_weight, weight_value.data, input_port_id=1)
        )

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode,
        scale_value: torch.Tensor,
        source_output_port_id: int,
        nodes: List[NNCFNode],
        scale_node_name: str,
    ) -> OVMultiplyInsertionCommand:
        input_port_id = 0
        target_points = []
        for node in nodes:
            target_points.append(PTTargetPoint(PT_PRE_LAYER_TARGET_TYPE, node.node_name, input_port_id=input_port_id))

        sq_multiply = FXSQMultiply(scale_value)
        return FXApplyTransformationCommand(
            module_insertion_transformation_builder(sq_multiply, target_points, scale_node_name)
        )

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
        # TODO(dlyakhov): Support shared layers in TorchFX.
        # Ref: 149316
        return False

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int, algorithm_key: str) -> Callable[[StatisticPoint], bool]:
        def filter_func(point: StatisticPoint) -> bool:
            return (
                algorithm_key in point.algorithm_to_tensor_collectors
                and point.target_point.type == PT_PRE_LAYER_TARGET_TYPE
                and point.target_point.input_port_id == activation_port_id
            )

        return filter_func
