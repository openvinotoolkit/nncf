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

import onnx

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.onnx.graph.model_utils import remove_fq_from_inputs
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_any_weight_quantized
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.onnx_helper import get_name_to_node_map
from nncf.onnx.graph.transformations.command_creation import create_bias_correction_command
from nncf.onnx.graph.transformations.commands import ONNXInitializerUpdateCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.collectors import get_mean_statistic_collector
from nncf.onnx.statistics.collectors import get_raw_stat_collector
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.tensor import Tensor


class ONNXBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
    ) -> ONNXInitializerUpdateCommand:
        return create_bias_correction_command(node, bias_value.data)

    @staticmethod
    def model_extraction_command(
        input_ids: Set[Tuple[str, int]], output_ids: Set[Tuple[str, int]]
    ) -> ONNXModelExtractionCommand:
        return ONNXModelExtractionCommand(input_ids, output_ids)

    @staticmethod
    def output_insertion_command(nncf_graph: NNCFGraph, target_point: ONNXTargetPoint) -> ONNXOutputInsertionCommand:
        nncf_input_node_next_nodes = {}
        for input_node in nncf_graph.get_input_nodes():
            next_nodes = nncf_graph.get_next_nodes(input_node)
            nncf_input_node_next_nodes[input_node.node_name] = [node.node_name for node in next_nodes]
        return ONNXOutputInsertionCommand(target_point, nncf_input_node_next_nodes)

    @staticmethod
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_statistic_collector(num_samples, channel_axis, window_size, inplace)

    @staticmethod
    def raw_statistic_collector(num_samples: int = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> Tensor:
        return Tensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[int, int]:
        return 0

    @staticmethod
    def get_bias_value(node: NNCFNode, model: onnx.ModelProto, nncf_graph: NNCFGraph) -> Tensor:
        return Tensor(get_bias_value(node, model))

    @staticmethod
    def get_input_name(model: onnx.ModelProto, node_name: str, input_port_id: int) -> str:
        node_mapping = get_name_to_node_map(model)
        return node_mapping[node_name].input[input_port_id]

    @staticmethod
    def get_output_name(model: onnx.ModelProto, node_name: str, output_port_id: int) -> str:
        node_mapping = get_name_to_node_map(model)
        return node_mapping[node_name].output[output_port_id]

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_any_weight_quantized(node, nncf_graph)

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node)

    @staticmethod
    def remove_fq_from_inputs(model: onnx.ModelProto, nncf_graph: NNCFGraph) -> onnx.ModelProto:
        return remove_fq_from_inputs(model, nncf_graph)

    @staticmethod
    def get_port_id(target_point: ONNXTargetPoint) -> int:
        return target_point.port_id
