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

from copy import deepcopy
from typing import Dict, List, Tuple, Optional
import numpy as np
import onnx

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.utils.backend import BackendType
from nncf.common.utils.logger import logger as nncf_logger

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.openvino_native.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.graph.model_transformer import OVModelTransformer

from nncf.experimental.openvino_natve.hardware.config import ONNXHWConfig
from nncf.experimental.openvino_natve.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
from nncf.experimental.openvino_natve.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_natve.statistics.collectors import OVMinMaxStatisticCollector
from nncf.experimental.openvino_natve.graph.onnx_graph import ONNXGraph

from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.min_max.utils import QuantizerLayerParameters


@ALGO_BACKENDS.register(BackendType.OPENVINO_NATIVE)
class OVMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return GENERAL_WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def hw_fused_patterns(self) -> HWFusedPatterns:
        return ONNX_HW_FUSED_PATTERNS

    @property
    def hw_config(self) -> HWConfig:
        return ONNXHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def model_transformer(model: ov.Model) -> OVModelTransformer:
        return OVModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def quantizer_insertion_command(target_point: OVTargetPoint,
                                    parameters: QuantizerLayerParameters) -> OVQuantizerInsertionCommand:
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def minmax_statistic_collector(use_abs_max: bool,
                                   reduction_shape: ReductionShape,
                                   num_samples: int = None) -> OVMinMaxStatisticCollector:
        return OVMinMaxStatisticCollector(use_abs_max, reduction_shape, num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(use_per_sample_stats: bool,
                                        use_abs_max: bool,
                                        reduction_shape: ReductionShape,
                                        num_samples: int = None,
                                        window_size: int = None) -> OVMeanMinMaxStatisticCollector:
        return OVMeanMinMaxStatisticCollector(use_per_sample_stats,
                                              use_abs_max,
                                              reduction_shape,
                                              num_samples,
                                              window_size)

    @staticmethod
    def get_weight_tensor(model: onnx.ModelProto, node: NNCFNode) -> Tuple[str, np.ndarray]:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node.node_name)
        return onnx_graph.get_weight_tensor(node)

    @staticmethod
    def get_weight_tensor_port_id(model: onnx.ModelProto, node: NNCFNode) -> Optional[int]:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(node.node_name)
        weight_tensor_name, _ = onnx_graph.get_weight_tensor(node)
        for i, input_name in enumerate(node.input):
            if input_name == weight_tensor_name:
                return i
        return None

    @staticmethod
    def get_tensor_names(node: NNCFNode) -> Tuple[List[str], List[str]]:
        return node.layer_attributes.input_tensor_names, \
               node.layer_attributes.output_tensor_names

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: onnx.ModelProto) -> QuantizerConfig:
        config = deepcopy(config)
        if model.opset_import[0].version < 13:
            config.per_channel = False
            nncf_logger.warning(
                f"Model opset version is {model.opset_import[0].version} < 13. "
                "Per-channel quantization is not supported. "
                "Set weight_quantizer_config.per_channel = False"
            )

        return config
