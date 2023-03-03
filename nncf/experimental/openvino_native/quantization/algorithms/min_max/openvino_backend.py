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

from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType

from nncf.experimental.openvino_native.graph.nncf_graph_builder import OVConstantLayerAttributes
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import GENERAL_WEIGHT_LAYER_METATYPES
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVTopKMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVNonMaxSuppressionMetatype
from nncf.experimental.openvino_native.graph.metatypes.openvino_metatypes import OVShapeMetatype
from nncf.experimental.openvino_native.graph.transformations.commands import OVQuantizerInsertionCommand
from nncf.experimental.openvino_native.graph.transformations.commands import OVTargetPoint
from nncf.experimental.openvino_native.hardware.config import OVHWConfig
from nncf.experimental.openvino_native.quantization.default_quantization import DEFAULT_OV_QUANT_TRAIT_TO_OP_DICT
from nncf.experimental.openvino_native.statistics.collectors import OVMeanMinMaxStatisticCollector
from nncf.experimental.openvino_native.statistics.collectors import OVMinMaxStatisticCollector
from nncf.experimental.openvino_native.quantization.quantizer_parameters import calculate_quantizer_parameters

from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS


@ALGO_BACKENDS.register(BackendType.OPENVINO)
class OVMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return GENERAL_WEIGHT_LAYER_METATYPES

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return [OVTopKMetatype, OVNonMaxSuppressionMetatype]

    @property
    def shape_of_metatypes(self) -> List[OperatorMetatype]:
        return [OVShapeMetatype]

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
    def create_activation_quantizer_insertion_command(
            nncf_graph: NNCFGraph,
            target_point: OVTargetPoint,
            quantizer_config: QuantizerConfig,
            statistics: MinMaxTensorStatistic) -> OVQuantizerInsertionCommand:
        parameters = calculate_quantizer_parameters(statistics, quantizer_config,
                                                    QuantizerGroup.ACTIVATIONS)
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def create_weight_quantizer_insertion_command(
            nncf_graph: NNCFGraph,
            target_point: OVTargetPoint,
            quantizer_config: QuantizerConfig,
            statistics: MinMaxTensorStatistic) -> OVQuantizerInsertionCommand:
        parameters = calculate_quantizer_parameters(statistics, quantizer_config,
                                                    QuantizerGroup.WEIGHTS)
        return OVQuantizerInsertionCommand(target_point, parameters)

    @staticmethod
    def _get_reduction_shape_and_use_abs_max(
            nncf_graph: NNCFGraph,
            target_point: OVTargetPoint,
            quantizer_config: QuantizerConfig) -> Tuple[ReductionShape, bool]:
        use_abs_max = quantizer_config.mode == QuantizationMode.SYMMETRIC
        if not quantizer_config.per_channel:
            return None, use_abs_max

        node = nncf_graph.get_node_by_name(target_point.target_node_name)
        if not target_point.is_weight_target_point():
            if target_point.type == TargetType.PRE_LAYER_OPERATION:
                shape = nncf_graph.get_input_edges(node)[target_point.port_id].tensor_shape
            elif target_point.type == TargetType.POST_LAYER_OPERATION:
                shape = nncf_graph.get_output_edges(node)[target_point.port_id].tensor_shape
            else:
                raise NotImplementedError(f'Unsupported target point type {target_point.type}.')

            # TODO (l-bat): Disable quantizer propogation through layout changing operations
            channel_axis = 1  # OpenVINO activations have channel first layout: [N, C, Z, Y, X]
            axes = tuple(i for i in range(len(shape)) if i != channel_axis)
            return axes, use_abs_max

        assert isinstance(node.layer_attributes, OVConstantLayerAttributes)
        const_shape = node.layer_attributes.const_shape

        if quantizer_config.per_channel:
            assert node.metatype in GENERAL_WEIGHT_LAYER_METATYPES
            channel_axis = node.metatype.const_channel_axis
            axes = tuple(i for i in range(len(const_shape)) if i not in channel_axis)
        else:
            axes = tuple(range(len(const_shape)))

        return axes, use_abs_max

    @staticmethod
    def minmax_statistic_collector(nncf_graph: NNCFGraph,
                                   target_point: OVTargetPoint,
                                   quantizer_config: QuantizerConfig,
                                   num_samples: int = None) -> OVMinMaxStatisticCollector:
        reduction_shape, use_abs_max =\
            OVMinMaxAlgoBackend._get_reduction_shape_and_use_abs_max(nncf_graph, target_point,
                                                                     quantizer_config)
        return OVMinMaxStatisticCollector(use_abs_max, reduction_shape, num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(nncf_graph: NNCFGraph,
                                        target_point: OVTargetPoint,
                                        quantizer_config: QuantizerConfig,
                                        use_per_sample_stats: bool,
                                        num_samples: int = None) -> OVMeanMinMaxStatisticCollector:
        reduction_shape, use_abs_max =\
            OVMinMaxAlgoBackend._get_reduction_shape_and_use_abs_max(nncf_graph, target_point,
                                                                     quantizer_config)
        return OVMeanMinMaxStatisticCollector(use_per_sample_stats,
                                              use_abs_max,
                                              reduction_shape,
                                              num_samples)

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        return node.layer_attributes.const_port_id
