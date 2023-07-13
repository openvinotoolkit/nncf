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

from typing import Dict, List, Optional, Tuple

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.graph.metatypes.common import FAKE_QUANTIZE_OPERATIONS
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.openvino.statistics.collectors import get_mean_stat_collector
from nncf.openvino.tensor import OVNNCFTensor
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):
    @property
    def tensor_processor(self) -> OVNNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: np.ndarray, nncf_graph: NNCFGraph
    ) -> OVBiasCorrectionCommand:
        return OVCommandCreator.create_command_to_update_bias(node, bias_value, nncf_graph)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> OVModelExtractionCommand:
        return OVModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(
        reduction_shape: ReductionShape,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_stat_collector(num_samples, reduction_shape, window_size, inplace)

    @staticmethod
    def get_sub_input_output_names(subgraph: ov.Model) -> Tuple[str, str]:
        return subgraph.inputs[0].get_any_name(), subgraph.outputs[0].get_any_name()

    @staticmethod
    def create_input_data(
        shape: Tuple[int], data: List[np.ndarray], input_name: str, channel_axis: int
    ) -> Dict[str, np.ndarray]:
        blob = np.zeros(shape)
        for j, idx in enumerate(np.ndindex(blob.shape[channel_axis])):
            index = tuple(slice(None) if i != channel_axis else idx for i in range(blob.ndim))
            blob[index] = data[j]
        blob = blob.astype(data[0].dtype)
        input_data = {input_name: blob}
        return input_data

    @staticmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
        return get_bias_value(node, nncf_graph, model)

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        # At first, checks whether the node has weight tensor
        if node.layer_attributes is None:
            return False
        const_port_ids = node.layer_attributes.get_const_port_ids()
        assert len(const_port_ids) == 1
        weight_node = nncf_graph.get_input_edges(node)[const_port_ids[0]].from_node
        return weight_node.metatype in FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> OVNNCFTensor:
        return OVNNCFTensor(raw_data[output_name])

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def get_bias_shift_magnitude(current_bias_value: np.ndarray, updated_bias_value: np.ndarray) -> float:
        bias_shift_magnitude = np.inf
        if np.count_nonzero(current_bias_value == 0) == 0:
            bias_shift_magnitude = np.max(np.abs((updated_bias_value - current_bias_value) / current_bias_value))
        return bias_shift_magnitude

    @staticmethod
    def post_process_output_data(data: List[np.ndarray]) -> np.ndarray:
        return np.array(data)

    @staticmethod
    def reshape_tensor(data: np.ndarray, new_shape: List[int]) -> np.ndarray:
        return data.reshape(new_shape)

    @staticmethod
    def get_node_names_for_input_output_statistics(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        return node.node_name, node.node_name
