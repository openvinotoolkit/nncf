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

from typing import Dict, List, Optional, Tuple

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.groups import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.metatypes.groups import OPERATIONS_WITH_BIAS_REDUCED
from nncf.openvino.graph.model_builder import OVModelBuilder
from nncf.openvino.graph.node_utils import get_activation_channel_axis
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import get_mean_statistic_collector
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.tensor import Tensor


class OVFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):

    def __init__(self, model):
        # Node mapping caching to reduce time for calculations
        self._node_mapping = {op.get_friendly_name(): op for op in model.get_ops()}
        self._model_builder = OVModelBuilder()

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
        input_ids: List[Tuple[str, int]], output_ids: List[Tuple[str, int]]
    ) -> OVModelExtractionCommand:
        return OVModelExtractionCommand(input_ids, output_ids)

    @staticmethod
    def mean_statistic_collector(
        channel_axis: int,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_statistic_collector(num_samples, channel_axis, window_size, inplace)

    @staticmethod
    def get_sub_input_output_names(subgraph: ov.Model) -> Tuple[str, str]:
        return subgraph.inputs[0].get_any_name(), subgraph.outputs[0].get_any_name()

    @staticmethod
    def create_input_data(
        shape: Tuple[int], data: List[Tensor], input_name: str, channel_axis: int
    ) -> Dict[str, np.ndarray]:
        blob = np.zeros(shape, dtype=data[0].data.dtype)
        for j, idx in enumerate(np.ndindex(blob.shape[channel_axis])):
            index = tuple(slice(None) if i != channel_axis else idx for i in range(blob.ndim))
            blob[index] = data[j].data
        input_data = {input_name: blob}
        return input_data

    def get_bias_value(self, node: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> Tensor:
        return Tensor(get_bias_value(node, nncf_graph, model, node_mapping=self._node_mapping))

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        activation_ports = [0, 1]

        for weight_port in node.layer_attributes.get_const_port_ids():
            activation_ports.remove(weight_port)
        assert len(activation_ports) == 1
        activation_port = activation_ports[0]

        return activation_port, 0

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        # At first, checks whether the node has weight tensor
        if node.layer_attributes is None:
            return False
        const_port_ids = node.layer_attributes.get_const_port_ids()
        assert len(const_port_ids) == 1
        weight_node = nncf_graph.get_input_edge_by_port_id(node, const_port_ids[0]).from_node
        return weight_node.metatype in FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> Tensor:
        return Tensor(raw_data[output_name])

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph, OPERATIONS_WITH_BIAS_REDUCED)

    @staticmethod
    def get_node_names_for_input_output_statistics(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        return node.node_name, node.node_name

    @staticmethod
    def get_activation_channel_axis(node: NNCFNode, port_id: int, input_shape: Tuple[int]) -> int:
        return get_activation_channel_axis(node, port_id, input_shape)

    def extract_submodel(self, model_transformer, input_id, output_id):

        return self._model_builder.build(
            input_ids=[input_id],
            output_ids=[output_id],
            node_mapping=self._node_mapping,
        )
