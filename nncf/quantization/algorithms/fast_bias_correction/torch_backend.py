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

from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import torch
from torch.quantization.fake_quantize import FakeQuantize

from nncf.common.graph import NNCFGraph
from nncf.common.graph import NNCFNode
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.registry import Registry
from nncf.quantization.algorithms.fast_bias_correction.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.fast_bias_correction.backend import FastBiasCorrectionAlgoBackend
from nncf.torch.graph.node_utils import get_bias_value
from nncf.torch.graph.node_utils import is_node_with_bias
from nncf.torch.graph.operator_metatypes import PT_OPERATOR_METATYPES
from nncf.torch.graph.transformations.command_creation import create_bias_correction_command
from nncf.torch.graph.transformations.commands import PTBiasCorrectionCommand
from nncf.torch.graph.transformations.commands import PTModelExtractionCommand
from nncf.torch.graph.transformations.commands import PTTargetPoint
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.tensor import PTNNCFTensor
from nncf.torch.tensor_statistics.collectors import PTMeanStatisticCollector
from nncf.torch.tensor_statistics.collectors import PTNNCFCollectorTensorProcessor


@ALGO_BACKENDS.register(BackendType.TORCH)
class PTFastBiasCorrectionAlgoBackend(FastBiasCorrectionAlgoBackend):

    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }

    @property
    def operation_metatypes(self) -> Registry:
        return PT_OPERATOR_METATYPES

    @property
    def tensor_processor(self) -> PTNNCFCollectorTensorProcessor:
        return PTNNCFCollectorTensorProcessor()

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in PTFastBiasCorrectionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = PTFastBiasCorrectionAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def create_bias_correction_command(node: NNCFNode,
                                       bias_value: np.ndarray,
                                       nncf_graph: NNCFGraph) -> PTBiasCorrectionCommand:
        return create_bias_correction_command(node, bias_value)

    @staticmethod
    def model_extraction_command(inputs: List[str], outputs: List[str]) -> PTModelExtractionCommand:
        return PTModelExtractionCommand(inputs[0])

    @staticmethod
    def mean_statistic_collector(reduction_shape: ReductionShape,
                                 num_samples: Optional[int] = None,
                                 window_size: Optional[int] = None) -> PTMeanStatisticCollector:
        return PTMeanStatisticCollector(reduction_shape, num_samples, window_size)

    @staticmethod
    def get_sub_input_output_names(subgraph: NNCFNetwork) -> Tuple[str, str]:
        # Pytorch does not have name for extracted node
        return None, None

    @staticmethod
    def create_input_data(shape: Tuple[int], data: List[np.ndarray], input_name: str, channel_axis: int) -> np.ndarray:
        blob = torch.zeros(shape, dtype=data[0].dtype)
        for j, idx in enumerate(np.ndindex(blob.shape[channel_axis])):
            index = tuple(slice(None) if i != channel_axis else idx for i in range(blob.ndim))
            blob[index] = data[j]
        return blob

    @staticmethod
    def get_bias_value(node: NNCFNode, nncf_graph: NNCFGraph, model: NNCFNetwork) -> np.ndarray:
        return get_bias_value(node, model)

    @staticmethod
    def get_activation_port_ids_for_bias_node(node: NNCFNode) -> Tuple[int, int]:
        return 0, 0

    @staticmethod
    def process_model_output(raw_data: Dict, output_name: str) -> PTNNCFTensor:
        return PTNNCFTensor(raw_data)

    @staticmethod
    def is_quantized_weights(node: NNCFNode, nncf_graph: NNCFGraph, model: NNCFNetwork) -> bool:
        node_module = model.get_containing_module(node.node_name)
        for pre_op in node_module.pre_ops.values():
            if isinstance(pre_op.op, (BaseQuantizer, FakeQuantize)):
                return True
        return False

    @staticmethod
    def is_node_with_bias(node: NNCFNode, nncf_graph: NNCFGraph, model: NNCFNetwork) -> bool:
        return is_node_with_bias(node, model)

    @staticmethod
    def get_bias_shift_magnitude(current_bias_value: torch.Tensor, updated_bias_value: torch.Tensor) -> float:
        bias_shift_magnitude = torch.inf
        if torch.count_nonzero(current_bias_value == 0) == 0:
            bias_shift_magnitude = torch.max(torch.abs((updated_bias_value - current_bias_value) / current_bias_value))
        return bias_shift_magnitude

    @staticmethod
    def post_process_output_data(data: List[torch.Tensor]) -> torch.Tensor:
        return torch.Tensor(data)
