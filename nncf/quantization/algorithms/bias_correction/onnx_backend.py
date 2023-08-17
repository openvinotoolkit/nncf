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
import onnx

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.onnx.graph.model_utils import remove_fq_from_inputs
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.node_utils import is_any_weight_quantized
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.onnx_graph import ONNXGraph
from nncf.onnx.graph.transformations.command_creation import create_bias_correction_command
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXNullBiasInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXOutputInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.onnx.statistics.collectors import ONNXNNCFCollectorTensorProcessor
from nncf.onnx.statistics.collectors import ONNXRawStatisticCollector
from nncf.onnx.tensor import ONNXNNCFTensor
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend


# pylint:disable=too-many-public-methods
@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):
    @property
    def tensor_processor(self) -> ONNXNNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    @property
    def types_to_insert_bias(self):
        return []

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(
        node: NNCFNode, bias_value: np.ndarray, nncf_graph: NNCFGraph
    ) -> ONNXBiasCorrectionCommand:
        return create_bias_correction_command(node, bias_value)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> ONNXModelExtractionCommand:
        return ONNXModelExtractionCommand(inputs, outputs)

    @staticmethod
    def create_bias_insertion_command(node: NNCFNode) -> ONNXNullBiasInsertionCommand:
        return ONNXNullBiasInsertionCommand(node)

    @staticmethod
    def output_insertion_command(nncf_graph: NNCFGraph, target_point: ONNXTargetPoint) -> ONNXOutputInsertionCommand:
        nncf_input_node_next_nodes = {}
        for input_node in nncf_graph.get_input_nodes():
            next_nodes = nncf_graph.get_next_nodes(input_node)
            nncf_input_node_next_nodes[input_node.node_name] = [node.node_name for node in next_nodes]
        return ONNXOutputInsertionCommand(target_point, nncf_input_node_next_nodes)

    @staticmethod
    def mean_statistic_collector(
        reduction_shape: ReductionShape,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> ONNXMeanStatisticCollector:
        return ONNXMeanStatisticCollector(reduction_shape, num_samples, window_size)

    @staticmethod
    def raw_statistic_collector(inplace: bool, num_samples: int = None) -> ONNXMeanStatisticCollector:
        return ONNXRawStatisticCollector(num_samples)

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_id(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[int, int]:
        return 0

    @staticmethod
    def get_bias_value(node: NNCFNode, model: onnx.ModelProto, nncf_graph: NNCFGraph) -> np.ndarray:
        return get_bias_value(node, model)

    @staticmethod
    def get_input_name(model: onnx.ModelProto, node_name: str) -> str:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node_name)
        return node.input[0]

    @staticmethod
    def get_output_name(model: onnx.ModelProto, node_name: str, output_id: int) -> List[str]:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node_name)
        return node.output[output_id]

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
    def insert_null_biases(model: onnx.ModelProto, nncf_graph: NNCFGraph) -> onnx.ModelProto:
        return model
