"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph, NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATOR_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVFakeQuantizeMetatype
from nncf.experimental.openvino_native.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVModelExtractionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.statistics.collectors import OVMeanStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.experimental.openvino_native.graph.node_utils import get_bias_value
from nncf.experimental.openvino_native.graph.node_utils import is_node_with_bias
from nncf.experimental.openvino_native.graph.transformations.command_creation import create_bias_correction_command
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):

    @property
    def operation_metatypes(self) -> Registry:
        return OV_OPERATOR_METATYPES

    @property
    def channel_axis_by_types(self) -> Dict[OVOpMetatype, int]:
        return {
            OVConvolutionMetatype: 1,
            OVConvolutionBackpropDataMetatype: 1,
            OVMatMulMetatype: -1
        }

    @property
    def tensor_processor(self) -> OVNNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(node: NNCFNode,
                                       bias_value: np.ndarray,
                                       nncf_graph: NNCFGraph) -> OVBiasCorrectionCommand:
        return create_bias_correction_command(node, bias_value, nncf_graph)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> OVModelExtractionCommand:
        return OVModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: Optional[int] = None,
                                 window_size: Optional[int] = None) -> OVMeanStatisticCollector:
        return OVMeanStatisticCollector(reduction_shape, num_samples, window_size)

    @staticmethod
    def get_sub_input_output_names(subgraph: ov.Model) -> Tuple[str, str]:
        return subgraph.inputs[0].node.friendly_name, subgraph.outputs[0].node.friendly_name

    @staticmethod
    def create_blob(shape: Tuple[int], data: List[float]) -> np.ndarray:
        blob = np.zeros(shape)
        for i, value in enumerate(data):
            blob[:, i] = value
        blob = blob.astype(np.float32)
        return blob

    @staticmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
        return get_bias_value(node, nncf_graph, model)

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        const_port_id = node.layer_attributes.const_port_id
        weight_node = nncf_graph.get_input_edges(node)[const_port_id].from_node
        return weight_node.metatype == OVFakeQuantizeMetatype

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> OVNNCFTensor:
        return OVNNCFTensor(raw_data[output_name])

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)
