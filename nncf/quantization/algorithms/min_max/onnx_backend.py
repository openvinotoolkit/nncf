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

from copy import deepcopy
from typing import Dict, List, Tuple
import numpy as np
import onnx

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.logging import nncf_logger

from nncf.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXNonMaxSuppressionMetatype
from nncf.onnx.graph.metatypes.onnx_metatypes import ONNXTopKMetatype
from nncf.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.onnx.hardware.config import ONNXHWConfig
from nncf.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
from nncf.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.onnx.quantization.quantizer_parameters import calculate_activation_quantizer_parameters
from nncf.onnx.quantization.quantizer_parameters import calculate_weight_quantizer_parameters
from nncf.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.onnx.statistics.collectors import ONNXMinMaxStatisticCollector
from nncf.onnx.graph.onnx_graph import ONNXGraph

from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS


@ALGO_BACKENDS.register(BackendType.ONNX)
class ONNXMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [ONNXTopKMetatype, ONNXNonMaxSuppressionMetatype]

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
    def model_transformer(model: onnx.ModelProto) -> ONNXModelTransformer:
        return ONNXModelTransformer(model)

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_activation_quantizer_insertion_command(target_point: ONNXTargetPoint,
                                                      quantizer_config: QuantizerConfig,
                                                      statistics: MinMaxTensorStatistic) \
                                                      -> ONNXQuantizerInsertionCommand:
        axis = 1 if quantizer_config.per_channel else None
        parameters = calculate_activation_quantizer_parameters(statistics, quantizer_config, axis)
        return ONNXQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def create_weight_quantizer_insertion_command(target_point: ONNXTargetPoint,
                                                  quantizer_config: QuantizerConfig,
                                                  weight_tensor: np.ndarray,
                                                  node: NNCFNode) -> ONNXQuantizerInsertionCommand:
        axis = node.metatype.weight_definitions.weight_channel_axis if quantizer_config.per_channel else None
        parameters = calculate_weight_quantizer_parameters(weight_tensor, quantizer_config, axis)
        return ONNXQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def minmax_statistic_collector(use_abs_max: bool,
                                   reduction_shape: ReductionShape,
                                   num_samples: int = None) -> ONNXMinMaxStatisticCollector:
        return ONNXMinMaxStatisticCollector(use_abs_max, reduction_shape, num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(use_per_sample_stats: bool,
                                        use_abs_max: bool,
                                        reduction_shape: ReductionShape,
                                        num_samples: int = None,
                                        window_size: int = None) -> ONNXMeanMinMaxStatisticCollector:
        return ONNXMeanMinMaxStatisticCollector(use_per_sample_stats,
                                                use_abs_max,
                                                reduction_shape,
                                                num_samples,
                                                window_size)

    @staticmethod
    def get_weight_tensor(model: onnx.ModelProto, target_point: ONNXTargetPoint) -> Tuple[str, np.ndarray]:
        onnx_graph = ONNXGraph(model)
        node = onnx_graph.get_node_by_name(target_point.target_node_name)
        return onnx_graph.get_weight_tensor(node)

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        return node.metatype.weight_definitions.weight_port_id

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: onnx.ModelProto) -> QuantizerConfig:
        config = deepcopy(config)
        if model.opset_import[0].version < 13:
            config.per_channel = False
            nncf_logger.warning(f"Model opset version is {model.opset_import[0].version} < 13 - "
                                "will not use per-channel quantization because it is not supported in this opset.")

        return config
