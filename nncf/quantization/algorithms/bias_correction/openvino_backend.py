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
from nncf.openvino.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.openvino.graph.node_utils import get_bias_value
from nncf.openvino.graph.node_utils import is_node_with_bias
from nncf.openvino.graph.transformations.command_creation import OVCommandCreator
from nncf.openvino.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.openvino.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.openvino.graph.transformations.commands import OVModelExtractionCommand
from nncf.openvino.graph.transformations.commands import OVOutputInsertionCommand
from nncf.openvino.graph.transformations.commands import OVTargetPoint
from nncf.openvino.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.openvino.statistics.collectors import get_mean_batch_stat_collector
from nncf.openvino.statistics.collectors import get_mean_stat_collector
from nncf.openvino.tensor import OVNNCFTensor
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend


# pylint:disable=too-many-public-methods
@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):
    @property
    def tensor_processor(self) -> OVNNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor

    @property
    def quantizer_types(self) -> List[OVOpMetatype]:
        return FAKE_QUANTIZE_OPERATIONS

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
    def output_insertion_command(nncf_graph: NNCFGraph, target_point: OVTargetPoint) -> OVOutputInsertionCommand:
        return OVOutputInsertionCommand(target_point)

    @staticmethod
    def node_removing_command(target_point: OVTargetPoint) -> OVFQNodeRemovingCommand:
        return OVFQNodeRemovingCommand(target_point)

    @staticmethod
    def mean_statistic_collector(
        reduction_shape: ReductionShape,
        inplace: bool,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> TensorCollector:
        return get_mean_stat_collector(num_samples, reduction_shape, window_size, inplace)

    @staticmethod
    def batch_statistic_collector(inplace: bool, num_samples: int = None) -> TensorCollector:
        return get_mean_batch_stat_collector(num_samples, inplace)

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> OVNNCFTensor:
        return OVNNCFTensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def get_bias_value(node: NNCFNode, model: ov.Model, nncf_graph: NNCFGraph) -> np.ndarray:
        return get_bias_value(node, nncf_graph, model)

    @staticmethod
    def get_input_name(model: ov.Model, node_name: str) -> str:
        ops_dict = {op.get_friendly_name(): op for op in model.get_ops()}

        model_input_names = []
        for tensor in model.inputs:
            model_input_names.extend(tensor.get_names())
        if node_name in model_input_names:
            return node_name

        for input_port in ops_dict[node_name].inputs():
            input_node = input_port.get_source_output().get_node()
            if input_node.get_type_name() == "Parameter":
                return input_port.get_tensor().get_any_name()
        raise RuntimeError(f"Input layer not found for {node_name}")

    @staticmethod
    def get_output_name(model: ov.Model, node_name: str) -> str:
        ops_dict = {op.get_friendly_name(): op for op in model.get_ops()}

        for output_port in ops_dict[node_name].outputs():
            for output_input_port in output_port.get_target_inputs():
                output_node = output_input_port.get_node()
                if output_node.get_type_name() == "Result":
                    return output_port.get_any_name()
        raise RuntimeError(f"Output layer not found for {node_name}")

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        const_port_ids = node.layer_attributes.get_const_port_ids()
        assert len(const_port_ids) == 1
        weight_node = nncf_graph.get_input_edges(node)[const_port_ids[0]].from_node
        return weight_node.metatype in FAKE_QUANTIZE_OPERATIONS

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)
