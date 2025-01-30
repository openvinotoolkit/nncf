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

import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.groups import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.model_utils import remove_fq_from_inputs
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import get_parameter_node_name
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import get_mean_statistic_collector
from nncf.openvino.statistics.collectors import get_raw_stat_collector
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.tensor import Tensor


class OVBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: Tensor, nncf_graph: NNCFGraph
    ) -> OVBiasCorrectionCommand:
        return OVCommandCreator.create_command_to_update_bias(node, bias_value.data, nncf_graph)

    @staticmethod
    def model_extraction_command(
        input_ids: Set[Tuple[str, int]], output_ids: Set[Tuple[str, int]]
    ) -> OVModelExtractionCommand:
        return OVModelExtractionCommand(input_ids, output_ids)

    @staticmethod
    def output_insertion_command(nncf_graph: NNCFGraph, target_point: OVTargetPoint) -> OVOutputInsertionCommand:
        return OVOutputInsertionCommand(target_point)

    @staticmethod
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_statistic_collector(num_samples, channel_axis, window_size, inplace)

    @staticmethod
    def raw_statistic_collector(num_samples: Optional[int] = None) -> TensorCollector:
        return get_raw_stat_collector(num_samples)

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> Tensor:
        return Tensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> int:
        constant_ports = node.layer_attributes.get_const_port_ids()
        activation_ports = [
            e.input_port_id for e in nncf_graph.get_input_edges(node) if e.input_port_id not in constant_ports
        ]
        assert len(activation_ports) == 1
        return activation_ports[0]

    @staticmethod
    def get_bias_value(node: NNCFNode, model: ov.Model, nncf_graph: NNCFGraph) -> Tensor:
        return Tensor(get_bias_value(node, nncf_graph, model))

    @staticmethod
    def get_input_name(model: ov.Model, node_name: str, input_port_id: int) -> str:
        return get_parameter_node_name(node_name, input_port_id)

    @staticmethod
    def get_output_name(model: ov.Model, node_name: str, output_port_id: int) -> str:
        return get_result_node_name(node_name, output_port_id)

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        if node.layer_attributes is None:
            return False
        const_port_ids = node.layer_attributes.get_const_port_ids()
        assert len(const_port_ids) == 1
        weight_node = nncf_graph.get_input_edge_by_port_id(node, const_port_ids[0]).from_node
        return weight_node.metatype in FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def remove_fq_from_inputs(model: ov.Model, nncf_graph: NNCFGraph) -> ov.Model:
        return remove_fq_from_inputs(model, nncf_graph)

    @staticmethod
    def get_port_id(target_point: OVTargetPoint) -> int:
        return target_point.port_id
