# Copyright (c) 2024 Intel Corporation
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
import torch
import torch.fx
from torch.ao.quantization.pt2e.utils import _get_tensor_constant_from_node

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.transformations import bias_update_transformation_builder
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.tensor import Tensor
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.tensor_statistics.collectors import get_mean_statistic_collector


class FXFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):
    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in FXFastBiasCorrectionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = FXFastBiasCorrectionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
    ) -> FXApplyTransformationCommand:
        return FXApplyTransformationCommand(bias_update_transformation_builder(node, bias_value.data))

    @staticmethod
    def model_extraction_command(
        input_ids: List[Tuple[str, int]], output_ids: List[Tuple[str, int]]
    ) -> PTModelExtractionCommand:
        return PTModelExtractionCommand([input_ids[0][0]], [output_ids[0][0]])

    @staticmethod
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_statistic_collector(num_samples, channel_axis, window_size)

    @staticmethod
    def get_sub_input_output_names(subgraph: NNCFNetwork) -> Tuple[str, str]:
        # Pytorch does not have name for extracted node
        return None, None

    @staticmethod
    def create_input_data(shape: Tuple[int], data: List[Tensor], input_name: str, channel_axis: int) -> torch.Tensor:
        blob = torch.zeros(shape, dtype=data[0].data.dtype, device=data[0].data.device)
        for j, idx in enumerate(np.ndindex(blob.shape[channel_axis])):
            index = tuple(slice(None) if i != channel_axis else idx for i in range(blob.ndim))
            blob[index] = data[j].data
        return blob

    @staticmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: torch.fx.GraphModule) -> Tensor:
        bias_node = nncf_graph.get_next_nodes(node)[0]
        # TODO(dlyakhov): make a node_name_vs_node map to speed up the process
        graph_bias_node = get_graph_node_by_name(model.graph, bias_node.node_name)
        return Tensor(_get_tensor_constant_from_node(graph_bias_node.all_input_nodes[1], model))

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> Tensor:
        return Tensor(raw_data)

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        weight_node = nncf_graph.get_previous_nodes(node)[1]
        return "dequantize" in weight_node.node_type

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        # Assumes that all biases were unfused
        if node.metatype in (om.PTConv1dMetatype, om.PTConv2dMetatype, om.PTConv3dMetatype, om.PTLinearMetatype):
            next_nodes = nncf_graph.get_next_nodes(node)
            if len(next_nodes) != 1:
                return False
            return next_nodes[0].metatype in (om.PTAddMetatype,)

    @staticmethod
    def get_node_names_for_input_output_statistics(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        return node.node_name, node.node_name

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, pord_id: int, input_shape: Tuple[int]) -> int:
        return node.metatype.output_channel_axis
