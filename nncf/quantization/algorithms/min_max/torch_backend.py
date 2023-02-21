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

from nncf.common.hardware.config import HWConfig
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType
from nncf.common.graph.transformations.commands import TransformationPriority

from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import get_scale_shape
from nncf.torch.quantization.init_range import PTRangeInitCollectorParams
from nncf.torch.quantization.init_range import StatCollectorGenerator
from nncf.torch.nncf_network import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.tensor_statistics.collectors import PTMinMaxStatisticCollector
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.collectors import PTMeanMinMaxStatisticCollector

from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS

import nncf.torch.graph.operator_metatypes as om

@ALGO_BACKENDS.register(BackendType.TORCH)
class PTMinMaxAlgoBackend(MinMaxAlgoBackend):

    @property
    def layers_with_weights_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.PTModuleConv1dMetatype,
            om.PTModuleConv2dMetatype,
            om.PTModuleConv3dMetatype,
            om.PTDepthwiseConv1dSubtype,
            om.PTDepthwiseConv2dSubtype,
            om.PTDepthwiseConv3dSubtype,
            om.PTModuleLinearMetatype,
            om.PTModuleConvTranspose1dMetatype,
            om.PTModuleConvTranspose2dMetatype,
            om.PTModuleConvTranspose3dMetatype,
            om.PTModuleEmbeddingMetatype,
            om.PTModuleEmbeddingBagMetatype,
        ]

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def hw_config(self) -> HWConfig:
        return PTHWConfig

    @property
    def quant_trait_op_dict(self) -> Dict[int, OperatorMetatype]:
        return DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT

    @staticmethod
    def model_transformer(model: NNCFNetwork) -> ModelTransformer:
        return PTModelTransformer(model)

    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }

    @staticmethod
    def target_point(target_type: TargetType,
                     target_node_name: str,
                     port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or\
            target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in PTMinMaxAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = PTMinMaxAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def create_activation_quantizer_insertion_command(
            nncf_graph: NNCFGraph,
            target_point: PTTargetPoint,
            quantizer_config: QuantizerConfig,
            statistics: PTMinMaxTensorStatistic) -> PTInsertionCommand:
        return PTMinMaxAlgoBackend._create_quantizer_insertion_command(nncf_graph,
                                                                       target_point,
                                                                       quantizer_config,
                                                                       statistics)
    @staticmethod
    def create_weight_quantizer_insertion_command(
            nncf_graph: NNCFGraph,
            target_point: PTTargetPoint,
            quantizer_config: QuantizerConfig,
            statistics: MinMaxTensorStatistic) -> PTInsertionCommand:
        return PTMinMaxAlgoBackend._create_quantizer_insertion_command(nncf_graph,
                                                                       target_point,
                                                                       quantizer_config,
                                                                       statistics)

    @staticmethod
    def minmax_statistic_collector(nncf_graph: NNCFGraph,
                                   target_point: PTTargetPoint,
                                   quantizer_config: QuantizerConfig,
                                   num_samples: int = None) -> PTMinMaxStatisticCollector:
        return PTMinMaxAlgoBackend._statistic_collector_builder("min_max",
                                                                nncf_graph,
                                                                target_point,
                                                                quantizer_config,
                                                                num_samples)

    @staticmethod
    def mean_minmax_statistic_collector(nncf_graph: NNCFGraph,
                                        target_point: PTTargetPoint,
                                        quantizer_config: QuantizerConfig,
                                        use_per_sample_stats: bool,
                                        num_samples: int = None) -> PTMeanMinMaxStatisticCollector:
        return PTMinMaxAlgoBackend._statistic_collector_builder("mean_min_max",
                                                                nncf_graph,
                                                                target_point,
                                                                quantizer_config,
                                                                num_samples)

    @staticmethod
    def get_weight_tensor_port_id(node: NNCFNode) -> int:
        return None

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: NNCFNetwork) -> QuantizerConfig:
        return config

    @staticmethod
    def _get_input_scale_shape(
            nncf_graph: NNCFGraph,
            target_point: PTTargetPoint,
            quantization_config: QuantizerConfig) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        is_weights = target_point.is_weight_target_point()
        if is_weights:
            module_node = nncf_graph.get_node_by_name(target_point.target_node_name)
            layer_attributes = module_node.layer_attributes
            assert isinstance(layer_attributes, WeightedLayerAttributes)
            input_shape = layer_attributes.get_weight_shape()
            channel_idx = layer_attributes.get_target_dim_for_compression()
        else:
            input_shape = nncf_graph.get_input_shape_for_insertion_point(target_point)
            channel_idx = 1  # channel dim for activations

        scale_shape = tuple(get_scale_shape(
                            input_shape,
                            is_weights=is_weights,
                            per_channel=quantization_config.per_channel,
                            channel_idx=channel_idx))

        return input_shape, scale_shape, channel_idx

    @staticmethod
    def _default_collector_params_and_scale_shape(
            nncf_graph: NNCFGraph,
            target_point: PTTargetPoint,
            quantizer_config: QuantizerConfig) -> Tuple[PTRangeInitCollectorParams, Tuple[int, ...]]:
        input_shape, scale_shape, channel_idx =\
            PTMinMaxAlgoBackend._get_input_scale_shape(nncf_graph, target_point, quantizer_config)
        return PTRangeInitCollectorParams(is_weights=target_point.is_weight_target_point(),
                                          mode=quantizer_config.mode,
                                          per_channel=quantizer_config.per_channel,
                                          input_shape=input_shape,
                                          channel_idx=channel_idx), scale_shape

    @staticmethod
    def _statistic_collector_builder(collector_name: str,
                                     nncf_graph: NNCFGraph,
                                     target_point: PTTargetPoint,
                                     quantizer_config: QuantizerConfig,
                                     num_samples: int = None) -> PTMeanMinMaxStatisticCollector:
        collector_params, scale_shape =\
            PTMinMaxAlgoBackend._default_collector_params_and_scale_shape(nncf_graph,
                                                                          target_point,
                                                                          quantizer_config)
        init_config = RangeInitConfig(collector_name, num_samples)
        return StatCollectorGenerator.generate_stat_collector_for_range_init_config(
            init_config,
            scale_shape,
            collector_params,
            num_samples)

    @staticmethod
    def _create_quantizer(quantizer_config: QuantizerConfig,
                          scale_shape: Tuple,
                          statistics: MinMaxTensorStatistic) -> BaseQuantizer:
        quantizer_cls = QUANTIZATION_MODULES.get(quantizer_config.mode)
        quantizer_spec = PTQuantizerSpec.from_config(quantizer_config,
                                                     narrow_range=False,
                                                     scale_shape=scale_shape,
                                                     half_range=False,
                                                     logarithm_scale=False,
                                                     is_quantized_on_export=False,
                                                     compression_lr_multiplier=None)
        quantizer = quantizer_cls(quantizer_spec)
        # Fill it with minmax
        quantizer.apply_minmax_init(min_values=statistics.min_values,
                                    max_values=statistics.max_values)
        return quantizer

    @staticmethod
    def _create_quantizer_insertion_command(nncf_graph: NNCFGraph,
                                            target_point: PTTargetPoint,
                                            quantizer_config: QuantizerConfig,
                                            statistics: MinMaxTensorStatistic) -> PTInsertionCommand:
        _, scale_shape, _ =\
            PTMinMaxAlgoBackend._get_input_scale_shape(nncf_graph, target_point, quantizer_config)
        quantizer = PTMinMaxAlgoBackend._create_quantizer(quantizer_config,
                                                          scale_shape, statistics)
        return PTInsertionCommand(target_point, quantizer, TransformationPriority.QUANTIZATION_PRIORITY)
