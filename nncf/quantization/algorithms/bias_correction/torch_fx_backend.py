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

from typing import Dict, Optional, Set, Tuple

import torch.fx

import nncf
from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.model_utils import get_target_point
from nncf.experimental.torch.fx.model_utils import remove_fq_from_inputs
from nncf.experimental.torch.fx.node_utils import get_bias_value
from nncf.experimental.torch.fx.node_utils import get_graph_node_by_name
from nncf.experimental.torch.fx.node_utils import is_node_with_bias
from nncf.experimental.torch.fx.transformations import constant_update_transformation_builder
from nncf.experimental.torch.fx.transformations import output_insertion_transformation_builder
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.tensor import Tensor
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.model_graph_manager import is_quantized_weights
from nncf.torch.tensor_statistics.collectors import get_mean_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_raw_stat_collector


class FXBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return get_target_point(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
    ) -> FXApplyTransformationCommand:
        return FXApplyTransformationCommand(
            constant_update_transformation_builder(node, bias_value.data, input_port_id=2)
        )

    @staticmethod
    def model_extraction_command(
        input_ids: Set[Tuple[str, int]], output_ids: Set[Tuple[str, int]]
    ) -> PTModelExtractionCommand:
        return PTModelExtractionCommand([inp_id[0] for inp_id in input_ids], [out_id[0] for out_id in output_ids])

    @staticmethod
    def output_insertion_command(nncf_graph: NNCFGraph, target_point: PTTargetPoint) -> FXApplyTransformationCommand:
        return FXApplyTransformationCommand(output_insertion_transformation_builder(target_point))

    @staticmethod
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_statistic_collector(num_samples, channel_axis, window_size)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: int) -> Tensor:
        return Tensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        return 0

    @staticmethod
    def get_bias_value(node: NNCFNode, model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> Tensor:
        return Tensor(get_bias_value(node, nncf_graph, model))

    @staticmethod
    def get_input_name(model: torch.fx.GraphModule, node_name: str, input_port_id: int) -> str:
        graph_node = get_graph_node_by_name(model.graph, node_name)
        return graph_node.all_input_nodes[input_port_id].name

    @staticmethod
    def get_output_name(model: torch.fx.GraphModule, node_name: str, output_port_id: int) -> int:
        graph_node = get_graph_node_by_name(model.graph, node_name)
        if graph_node.op == "output":
            # Original node output is kept as the first
            # output tensor, thus returns 0.
            return 0
        nodes = list(graph_node.users)
        while nodes:
            node = nodes.pop()
            if node.op == "call_function" and node.target == torch.ops.aten.clone.default:
                nodes = list(node.users)
                graph_node = node
            elif node.op == "output":
                return node.all_input_nodes.index(graph_node)

        raise nncf.InternalError(f"Node with name {node_name} expected to have an output," " no outputs were found.")

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_quantized_weights(node, nncf_graph)

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def remove_fq_from_inputs(model: torch.fx.GraphModule, nncf_graph: NNCFGraph) -> torch.fx.GraphModule:
        return remove_fq_from_inputs(model, nncf_graph)

    @staticmethod
    def get_port_id(target_point: PTTargetPoint) -> int:
        if target_point.target_type == TargetType.OPERATOR_POST_HOOK:
            # Return 0 as default value for post hook port id.
            return 0
        return target_point.input_port_id
