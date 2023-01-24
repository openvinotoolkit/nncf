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

from typing import Dict, Tuple, List, Optional
import onnx
import numpy as np
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph

from nncf.onnx.graph.metatypes.onnx_metatypes import ONNX_OPERATION_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXOpMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXConvolutionTransposeMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDepthwiseConvolutionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXDequantizeLinearMetatype
from nncf.onnx.graph.transformations.commands import ONNXBiasCorrectionCommand
from nncf.onnx.graph.transformations.commands import ONNXModelExtractionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.statistics.collectors import ONNXMeanStatisticCollector
from nncf.onnx.statistics.collectors import ONNXNNCFCollectorTensorProcessor
from nncf.onnx.tensor import ONNXNNCFTensor
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.onnx.graph.node_utils import is_node_with_bias
from nncf.onnx.graph.node_utils import get_bias_value
from nncf.onnx.graph.transformations.command_creation import create_bias_correction_command


@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):

    @property
    def operation_metatypes(self) -> Registry:
        return ONNX_OPERATION_METATYPES

    @property
    def channel_axis_by_types(self) -> Dict[ONNXOpMetatype, int]:
        return {
            ONNXConvolutionMetatype: 1,
            ONNXConvolutionTransposeMetatype: 1,
            ONNXDepthwiseConvolutionMetatype: 1
        }

    @property
    def tensor_processor(self) -> ONNXNNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(bias_node: NNCFNode, bias_value: np.ndarray) -> ONNXBiasCorrectionCommand:
        create_bias_correction_command(bias_node, bias_value)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> ONNXModelExtractionCommand:
        return ONNXModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: Optional[int] = None,
                                 window_size: Optional[int] = None) -> ONNXMeanStatisticCollector:
        return ONNXMeanStatisticCollector(reduction_shape, num_samples, window_size)

    @staticmethod
    def get_input_output_names(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        return biased_node.layer_attributes.input_tensor_names[0], \
               biased_node.layer_attributes.output_tensor_names[0]

    @staticmethod
    def create_blob(shape: Tuple[int], data: List[float]) -> np.ndarray:
        blob = np.zeros(shape)
        for i, value in enumerate(data):
            blob[:, i] = value
        blob = blob.astype(np.float32)
        return blob

    @staticmethod
    def get_bias_value(bias_node: NNCFNode, model: onnx.ModelProto) -> np.ndarray:
        return get_bias_value(bias_node, model)

    @staticmethod
    def get_activation_port_ids_for_bias_node(model: onnx.ModelProto, biased_node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> ONNXNNCFTensor:
        return ONNXNNCFTensor(raw_data[output_name])

    @staticmethod
    def is_quantized_weights(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        input_nodes = nncf_graph.get_previous_nodes(biased_node)
        weight_port_id = biased_node.metatype.weight_definitions.weight_port_id
        weight_node = input_nodes[weight_port_id]
        return weight_node.metatype == ONNXDequantizeLinearMetatype

    @staticmethod
    def is_node_with_bias(biased_node: NNCFNode) -> bool:
        return is_node_with_bias(biased_node)

    @staticmethod
    def get_bias_node(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
        return biased_node
