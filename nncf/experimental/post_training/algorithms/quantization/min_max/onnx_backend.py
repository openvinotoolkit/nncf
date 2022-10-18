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
from typing import Dict
from typing import List
from typing import Tuple
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

from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import WEIGHT_LAYER_METATYPES
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXNonMaxSuppressionMetatype
from nncf.experimental.onnx.graph.metatypes.onnx_metatypes import ONNXTopKMetatype
from nncf.experimental.onnx.graph.model_transformer import ONNXModelTransformer
from nncf.experimental.onnx.graph.transformations.commands import ONNXQuantizerInsertionCommand
from nncf.experimental.onnx.graph.transformations.commands import ONNXTargetPoint
from nncf.experimental.onnx.hardware.config import ONNXHWConfig
from nncf.experimental.onnx.hardware.fused_patterns import ONNX_HW_FUSED_PATTERNS
from nncf.experimental.onnx.quantization.default_quantization import DEFAULT_ONNX_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.onnx.statistics.collectors import ONNXMeanMinMaxStatisticCollector
from nncf.experimental.onnx.statistics.collectors import ONNXMinMaxStatisticCollector

from nncf.experimental.post_training.algorithms.quantization.min_max.backend import MinMaxAlgoBackend
from nncf.experimental.post_training.algorithms.quantization.min_max.backend import ALGO_BACKENDS
from nncf.experimental.post_training.algorithms.quantization.min_max.utils import QuantizerLayerParameters


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
                     target_node_name: str = None,
                     edge_name: str = None) -> ONNXTargetPoint:
        return ONNXTargetPoint(target_type, target_node_name, edge_name)

    @staticmethod
    def quantizer_insertion_command(target_point: ONNXTargetPoint,
                                    parameters: QuantizerLayerParameters) -> ONNXQuantizerInsertionCommand:
        return ONNXQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def minmax_statistic_collector(use_abs_max: bool,
                                   reduction_shape: ReductionShape,
                                   num_samples: int = None) -> ONNXMinMaxStatisticCollector:
        return ONNXMinMaxStatisticCollector(use_abs_max, reduction_shape,  num_samples)

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
    def get_initializer_value(model: onnx.ModelProto, initializer_name: str) -> np.ndarray:
        for initializer in model.graph.initializer:
            if initializer.name == initializer_name:
                return onnx.numpy_helper.to_array(initializer)
        raise RuntimeError(
            'There is no initializer with the name {}'.format(initializer_name))

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
