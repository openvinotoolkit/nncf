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

from typing import Dict, List, Tuple
import numpy as np
import openvino.runtime as ov

from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.patterns import HWFusedPatterns
from nncf.common.graph.transformations.commands import TargetPoint
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType

from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OV_OPERATOR_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVConvertMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.openvino_native.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.hardware.config import OVHWConfig
from nncf.experimental.openvino_native.hardware.fused_patterns import OPENVINO_HW_FUSED_PATTERNS
from nncf.experimental.openvino_native.quantization.default_quantization import DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.openvino_native.quantization.quantizer_parameters import calculate_activation_quantizer_parameters
from nncf.experimental.openvino_native.quantization.quantizer_parameters import calculate_weight_quantizer_parameters
from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector

from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return GENERAL_WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def hw_fused_patterns(self) -> HWFusedPatterns:
        return OPENVINO_HW_FUSED_PATTERNS

    @property
    def hw_config(self) -> HWConfig:
        return OVHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> OVTargetPoint:
        return OVTargetPoint(target_type, target_node_name, port_id)

    @staticmethod
    def create_activation_quantizer_insertion_command(target_point: OVTargetPoint,
                                                      quantizer_config: QuantizerConfig,
                                                      statistics: MinMaxTensorStatistic) \
                                                      -> OVQuantizerInsertionCommand:
        parameters = calculate_activation_quantizer_parameters(statistics, quantizer_config)
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def create_weight_quantizer_insertion_command(target_point: OVTargetPoint,
                                                  quantizer_config: QuantizerConfig,
                                                  weight_tensor: np.ndarray,
                                                  node: NNCFNode) -> OVQuantizerInsertionCommand:
        parameters = calculate_weight_quantizer_parameters(weight_tensor, quantizer_config, node.metatype)
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
    def get_weight_tensor(model: ov.Model, target_point: TargetPoint) -> Tuple[str, np.ndarray]:
        target_name = target_point.target_node_name
        for op in model.get_ops():
            if op.get_friendly_name() == target_name:
                node = op.input_value(target_point.port_id).get_node()
                # TODO(l-bat): Unify weights and activaions statistic collections. Add Result for weight nodes.
                metatype = OV_OPERATOR_METATYPES.get_operator_metatype_by_op_name(node.get_type_name())
                if metatype == OVConvertMetatype:
                    node = node.input_value(0).get_node()
                weight_tensor = node.get_vector().reshape(node.get_output_shape(0))
                return node.get_friendly_name(), weight_tensor
        raise RuntimeError(f'Could not find node: {target_name} in model.')

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        return node.layer_attributes.weight_port_id

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: ov.Model) -> QuantizerConfig:
        return config
