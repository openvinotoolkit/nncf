"""
 Copyright (c) 2022 Intel Corporation
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

from collections import deque
from typing import Dict, Tuple, List, Optional
import openvino.runtime as ov
import numpy as np
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.graph import NNCFNode
from nncf.common.graph import NNCFGraph
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.graph.operator_metatypes import OperatorMetatype

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATOR_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvolutionBackpropDataMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVMatMulMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVFakeQuantizeMetatype
from nncf.experimental.openvino_native.graph.model_transformer import OVModelTransformer
from nncf.experimental.openvino_native.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVModelExtractionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVFQNodeRemovingCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVOutputInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.statistics.collectors import OVMeanStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVBatchStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.quantization.algorithms.bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.bias_correction.backend import BiasCorrectionAlgoBackend
from nncf.experimental.openvino_native.graph.node_utils import get_bias_value
from nncf.experimental.openvino_native.graph.node_utils import is_node_with_bias
from nncf.experimental.openvino_native.graph.transformations.command_creation import create_bias_correction_command


#pylint:disable=too-many-public-methods
@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVBiasCorrectionAlgoBackend(BiasCorrectionAlgoBackend):

    @property
    def layers_with_bias_metatypes(self) -> List[OVOpMetatype]:
        return LAYERS_WITH_BIAS_METATYPES

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

    @property
    def quantizer_types(self) -> List[OperatorMetatype]:
        return [OVFakeQuantizeMetatype]

    @staticmethod
    def model_transformer(model: ov.Model) -> OVModelTransformer:
        return OVModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str = None,
                     port_id: str = None) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_bias_correction_command(node: NNCFNode,
                                       bias_value: np.ndarray,
                                       nncf_graph: NNCFGraph) -> OVBiasCorrectionCommand:
        return create_bias_correction_command(node, bias_value, nncf_graph)

    @staticmethod
    def output_insertion_command(target_point: OVTargetPoint) -> OVOutputInsertionCommand:
        return OVOutputInsertionCommand(target_point)

    @staticmethod
    def node_removing_command(target_point: OVTargetPoint) -> OVFQNodeRemovingCommand:
        return OVFQNodeRemovingCommand(target_point=target_point)

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: Optional[int] = None,
                                 window_size: Optional[int] = None) -> OVMeanStatisticCollector:
        return OVMeanStatisticCollector(reduction_shape,  num_samples, window_size)

    @staticmethod
    def batch_statistic_collector(num_samples: int = None) -> OVMeanStatisticCollector:
        return OVBatchStatisticCollector(num_samples)

    @staticmethod
    def get_input_name(node: NNCFNode) -> str:
        return node.node_name

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> OVNNCFTensor:
        return OVNNCFTensor(raw_data[output_name])

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: ov.Model) -> np.ndarray:
        return get_bias_value(node, nncf_graph, model)

    @staticmethod
    def get_bias_port_id(bias_node: NNCFNode) -> int:
        return bias_node.layer_attributes.weight_port_id

    @staticmethod
    def get_subgraph_output_name(model: ov.Model, node_name: str) -> str:
        for model_output in model.outputs:
            preview_node = model_output.node.input_value(0).get_node()
            if preview_node.get_friendly_name() == node_name:
                return model_output.node.get_friendly_name()
        return RuntimeError(f'Could not find the {node_name} output')

    @staticmethod
    def extract_model(model: ov.Model,
                      input_node_names: List[str],
                      output_node_names: List[str]) -> ov.Model:

        transformation_layout = TransformationLayout()
        model_transformer = OVModelTransformer(model)
        input_node_names = set(input_node_names)
        output_node_names = set(output_node_names)
        sub_input_names, sub_output_names = OVBiasCorrectionAlgoBackend.get_sub_input_output_names(input_node_names,
                                                                                                   output_node_names)
        inputs = (input_node_names, sub_input_names)
        outputs = (output_node_names, sub_output_names)
        transformation_layout.register(OVModelExtractionCommand(inputs, outputs))
        return model_transformer.transform(transformation_layout)

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        weight_port_id = node.layer_attributes.weight_port_id
        weight_node = nncf_graph.get_input_edges(node)[weight_port_id].from_node
        return weight_node.metatype == OVFakeQuantizeMetatype

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        return is_node_with_bias(node, nncf_graph)

    @staticmethod
    def get_sub_input_output_names(input_names: List[str], output_names: List[str]) -> Tuple[List[str], List[str]]:
        return [f'{name}_input' for name in input_names], [f'{name}_output' for name in output_names]
