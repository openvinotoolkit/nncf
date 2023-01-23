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
from typing import Dict, List, Tuple, Optional

import numpy as np
import openvino.runtime as ov

from nncf.common.graph import NNCFGraph, NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVOpMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import LAYERS_WITH_BIAS_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATOR_METATYPES
from nncf.experimental.openvino_native.graph.model_transformer import OVModelTransformer
from nncf.experimental.openvino_native.graph.transformations.commands import OVBiasCorrectionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVModelExtractionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
# TODO (KodiaqQ): Remove this WA after merging #1444
# pylint: disable=import-error,no-name-in-module
from nncf.experimental.openvino_native.statistics.collectors import OVMeanStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVNNCFCollectorTensorProcessor
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):

    @property
    def layers_with_bias_metatypes(self) -> List[OVOpMetatype]:
        return LAYERS_WITH_BIAS_METATYPES

    @property
    def channel_axis_by_types(self) -> Dict[OVOpMetatype, int]:
        return {
            OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('Convolution'): 1,
            OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('ConvolutionBackpropData'): 1,
            OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('MatMul'): -1
        }

    @property
    def tensor_processor(self) -> OVNNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    @staticmethod
    def model_transformer(model: ov.Model) -> OVModelTransformer:
        return OVModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def bias_correction_command(target_point: OVTargetPoint, bias_value: np.ndarray) -> OVBiasCorrectionCommand:
        return OVBiasCorrectionCommand(target_point, bias_value)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> OVModelExtractionCommand:
        return OVModelExtractionCommand(inputs, outputs)

    @staticmethod
    def mean_statistic_collector(reduction_shape:
                                 ReductionShape,
                                 num_samples: int = None,
                                 window_size: int = None) -> OVMeanStatisticCollector:
        return OVMeanStatisticCollector(reduction_shape, num_samples, window_size)

    @staticmethod
    def get_input_output_names(node: NNCFNode, nncf_graph: NNCFGraph) -> Tuple[str, str]:
        input_name = node.node_name
        output_name = node.node_name

        bias_node = OVFastBiasCorrectionAlgoBackend.get_bias_node(node, nncf_graph)
        if bias_node is not None:
            output_name = bias_node.node_name
        return input_name, output_name

    @staticmethod
    def create_blob(shape: Tuple[int], data: List[float]) -> np.ndarray:
        blob = np.zeros(shape)
        for i, value in enumerate(data):
            blob[:, i] = value
        blob = blob.astype(np.float32)
        return blob

    @staticmethod
    def get_bias_value(model: ov.Model, bias_node: NNCFNode) -> np.ndarray:
        ops_dict = {op.get_friendly_name(): op for op in model.get_ops()}

        bias_node = ops_dict[bias_node.node_name]
        skip_metatypes = ['Convert']
        needed_bias_type = 'Constant'

        potential_bias_queue = deque([n.node for n in bias_node.input_values()])
        while potential_bias_queue:
            potential_bias = potential_bias_queue.popleft()
            if potential_bias.get_type_name() in skip_metatypes:
                # We goes thorough 0 port in assumption that bias graph without branching
                potential_bias_queue.append(potential_bias.input_value(0).node)
            elif potential_bias.get_type_name() == needed_bias_type:
                bias_value = potential_bias.get_data()
                return bias_value.flatten(), bias_value.shape
            continue

        raise RuntimeError('Could not find the bias value of the node')

    @staticmethod
    def get_activation_port_ids_for_bias_node(model: ov.Model, biased_node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def is_quantized_weights(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        weight_port_id = biased_node.layer_attributes.weight_port_id
        weight_node = nncf_graph.get_input_edges(biased_node)[weight_port_id].from_node
        fq_type = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('FakeQuantize')
        return weight_node.metatype == fq_type

    @staticmethod
    def get_bias_port_id(bias_node: NNCFNode) -> int:
        return bias_node.layer_attributes.weight_port_id

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> OVNNCFTensor:
        return OVNNCFTensor(raw_data[output_name])

    @staticmethod
    def is_node_with_bias(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> bool:
        bias_node = OVFastBiasCorrectionAlgoBackend.get_bias_node(biased_node, nncf_graph)
        if bias_node is None:
            return False

        needed_bias_type = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('Constant')
        skip_metatypes = [OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('Convert')]

        potential_bias_queue = deque(nncf_graph.get_previous_nodes(bias_node))
        while potential_bias_queue:
            potential_bias = potential_bias_queue.popleft()
            if potential_bias.metatype in skip_metatypes:
                # We goes thorough 0 port in assumption that bias graph without branching
                potential_bias_queue.append(nncf_graph.get_previous_nodes(potential_bias)[0])
            elif potential_bias.metatype == needed_bias_type:
                return True
            continue
        return False

    @staticmethod
    def get_bias_node(biased_node: NNCFNode, nncf_graph: NNCFGraph) -> Optional[NNCFNode]:
        add_bias_node = nncf_graph.get_next_nodes(biased_node)[0]
        if add_bias_node.metatype == OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name('Add'):
            return add_bias_node
        return None
