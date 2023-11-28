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

from typing import List, Tuple

import numpy as np

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.quantizer_propagation.structs import QuantizationTrait
from nncf.common.tensor_statistics.statistic_point import StatisticPoint
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.tensor import Tensor
from nncf.openvino.graph.node_utils import get_channel_agnostic_reduction_axes
from nncf.openvino.graph.transformations.commands import OVMultiplyInsertionCommand
from nncf.openvino.graph.transformations.commands import OVWeightUpdateCommand
from nncf.quantization.algorithms.smooth_quant.backend import SmoothQuantAlgoBackend
from nncf.torch.graph.transformations.command_creation import create_command_to_update_weight
from nncf.torch.graph.transformations.command_creation import multiply_insertion_command
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.tensor_statistics.collectors import PTAbsMaxReducer
from nncf.torch.tensor_statistics.collectors import PTNNCFCollectorTensorProcessor


class PTSmoothQuantAlgoBackend(SmoothQuantAlgoBackend):
    @property
    def convolution_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.PTConv1dMetatype,
            om.PTConv2dMetatype,
            om.PTConv3dMetatype,
            om.PTModuleConv1dMetatype,
            om.PTModuleConv2dMetatype,
            om.PTModuleConv3dMetatype,
            om.PTDepthwiseConv1dSubtype,
            om.PTDepthwiseConv2dSubtype,
            om.PTDepthwiseConv3dSubtype,
            om.PTConvTranspose1dMetatype,
            om.PTConvTranspose2dMetatype,
            om.PTConvTranspose3dMetatype,
        ]

    @property
    def matmul_metatypes(self) -> List[OperatorMetatype]:
        return [om.PTMatMulMetatype, om.PTLinearMetatype, om.PTModuleLinearMetatype]

    @property
    def quantize_agnostic_metatypes(self) -> List[OperatorMetatype]:
        return DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT[QuantizationTrait.QUANTIZATION_AGNOSTIC]

    @staticmethod
    def target_point(target_node_name: str, port_id: int) -> PTTargetPoint:
        return PTTargetPoint(TargetType.OPERATOR_PRE_HOOK, target_node_name, input_port_id=port_id)

    @staticmethod
    def is_node_with_weights(node: NNCFNode) -> bool:
        return node.layer_attributes is not None

    @staticmethod
    def get_activations_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        return 0

    @staticmethod
    def get_channel_agnostic_reduction_axes(channel_axis: int, shape: Tuple[int]) -> Tuple[int]:
        return get_channel_agnostic_reduction_axes([channel_axis], shape)

    @staticmethod
    def get_abs_max_channel_collector(
        num_samples: int, stats_reduction_axes: Tuple[int], inplace: bool, branch_key: str
    ) -> TensorCollector:
        collector = TensorCollector()
        reducer = PTAbsMaxReducer(reduction_axes=stats_reduction_axes)
        aggregator = MaxAggregator(tensor_processor=PTNNCFCollectorTensorProcessor, num_samples=num_samples)
        collector.register_statistic_branch(branch_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_value(node_with_weight: NNCFNode, model: NNCFNetwork) -> Tensor:
        node_module = model.nncf.get_containing_module(node_with_weight.node_name)
        if node_module.weight is None:
            return None
        return Tensor(node_module.weight.data)

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        const_ids = node.layer_attributes.get_const_port_ids()
        if len(const_ids) != 1:
            raise RuntimeError(f"Found more than 1 port for {node.node_name} node")
        return const_ids[0]

    @staticmethod
    def weight_update_command(node_with_weight: NNCFNode, weight_value: np.ndarray) -> OVWeightUpdateCommand:
        return create_command_to_update_weight(node_with_weight, weight_value)

    @staticmethod
    def scale_insertion_command(
        source_node: NNCFNode,
        scale_value: np.ndarray,
        source_output_port_id: int,
        nodes: List[NNCFNode],
        scale_node_name: str,
    ) -> OVMultiplyInsertionCommand:
        input_port_id = 0
        return multiply_insertion_command(nodes, scale_value, scale_node_name, input_port_id)

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int) -> int:
        if node.metatype == om.PTModuleLinearMetatype:
            return -1
        # TODO: Add activation axis calculation when MatMul wiil be supported
        return 1

    @staticmethod
    def get_weight_channel_axis(node: NNCFNode) -> int:
        # TODO: Add activation axis calculation when MatMul wiil be supported
        return 1

    @staticmethod
    def is_node_with_shared_weight(node: NNCFNode, nncf_graph: NNCFGraph):
        return node.is_shared()

    @staticmethod
    def get_filter_fn_for_statistics(activation_port_id: int):
        def filter_func(point: StatisticPoint) -> bool:
            return True

        return filter_func
