# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch

import nncf.torch.graph.operator_metatypes as om
from nncf.common.graph.definitions import NNCFGraphNodeType
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.graph.model_transformer import ModelTransformer
from nncf.common.graph.operator_metatypes import OperatorMetatype
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.graph.transformations.commands import TransformationPriority
from nncf.common.hardware.config import HWConfig
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.utils.backend import BackendType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AggregatorType
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.quantization.algorithms.min_max.backend import ALGO_BACKENDS
from nncf.quantization.algorithms.min_max.backend import MinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.scopes import IgnoredScope
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTInsertionCommand
from nncf.torch.hardware.config import PTHWConfig
from nncf.torch.model_transformer import PTModelTransformer
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.default_quantization import DEFAULT_PT_QUANT_TRAIT_TO_OP_DICT
from nncf.torch.quantization.init_range import PTRangeInitCollectorParams
from nncf.torch.quantization.init_range import StatCollectorGenerator
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import get_scale_shape
from nncf.torch.tensor_statistics.collectors import PTMeanMinMaxStatisticCollector
from nncf.torch.tensor_statistics.collectors import PTMinMaxStatisticCollector
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic


# pylint:disable=too-many-public-methods
@ALGO_BACKENDS.register(BackendType.TORCH)
class PTMinMaxAlgoBackend(MinMaxAlgoBackend):
    TARGET_TYPE_TO_PT_INS_TYPE_MAP = {
        TargetType.PRE_LAYER_OPERATION: TargetType.OPERATOR_PRE_HOOK,
        TargetType.POST_LAYER_OPERATION: TargetType.OPERATOR_POST_HOOK,
    }

    @property
    def mat_mul_metatype(self) -> OperatorMetatype:
        return om.PTModuleLinearMetatype

    @property
    def post_processing_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def shapeof_metatypes(self) -> List[OperatorMetatype]:
        return []

    @property
    def conv_metatype(self) -> List[OperatorMetatype]:
        return [om.PTModuleConv1dMetatype, om.PTModuleConv2dMetatype, om.PTModuleConv3dMetatype]

    @property
    def overflow_fix_metatypes(self) -> List[OperatorMetatype]:
        return [
            om.PTModuleConv1dMetatype,
            om.PTModuleConv2dMetatype,
            om.PTModuleConv3dMetatype,
            om.PTModuleLinearMetatype,
            om.PTModuleConvTranspose1dMetatype,
            om.PTModuleConvTranspose2dMetatype,
            om.PTModuleConvTranspose3dMetatype,
        ]

    @property
    def read_variable_metatypes(self) -> List[OperatorMetatype]:
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

    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        if NNCFGraphNodeType.INPUT_NODE in target_node_name or target_type == TargetType.POST_LAYER_OPERATION:
            port_id = None
        if target_type in PTMinMaxAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP:
            target_type = PTMinMaxAlgoBackend.TARGET_TYPE_TO_PT_INS_TYPE_MAP[target_type]
        return PTTargetPoint(target_type, target_node_name, input_port_id=port_id)

    @staticmethod
    def create_activation_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> PTInsertionCommand:
        return PTMinMaxAlgoBackend._create_quantizer_insertion_command(
            nncf_graph, target_point, quantizer_config, parameters
        )

    @staticmethod
    def create_weight_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> PTInsertionCommand:
        return PTMinMaxAlgoBackend._create_quantizer_insertion_command(
            nncf_graph, target_point, quantizer_config, parameters
        )

    @staticmethod
    def unify_statistics(statistics: List[PTMinMaxTensorStatistic]) -> PTMinMaxTensorStatistic:
        max_values, min_values = [], []
        for statistic in statistics:
            max_values.append(statistic.max_values)
            min_values.append(statistic.min_values)
        max_values = torch.max(torch.tensor(max_values))
        min_values = torch.min(torch.tensor(min_values))
        return PTMinMaxTensorStatistic(min_values=min_values, max_values=max_values)

    @staticmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        inplace: bool,
        num_samples: int = None,
    ) -> Union[PTMinMaxStatisticCollector, PTMeanMinMaxStatisticCollector]:
        if (
            range_estimator_params.min.statistics_type == StatisticsType.MIN
            and range_estimator_params.min.aggregator_type == AggregatorType.MIN
            and range_estimator_params.max.statistics_type == StatisticsType.MAX
            and range_estimator_params.max.aggregator_type == AggregatorType.MAX
        ):
            collector_name = "min_max"

        elif (
            range_estimator_params.min.statistics_type == StatisticsType.MIN
            and range_estimator_params.min.aggregator_type == AggregatorType.MEAN
            and range_estimator_params.max.statistics_type == StatisticsType.MAX
            and range_estimator_params.max.aggregator_type == AggregatorType.MEAN
        ):
            collector_name = "mean_min_max"

        else:
            raise RuntimeError(
                "The following range estimator parameters are not supported by PyTorch backend by now: "
                f"{str(range_estimator_params)}"
            )

        return PTMinMaxAlgoBackend._statistic_collector_builder(
            collector_name, nncf_graph, target_point, quantizer_config, num_samples
        )

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode) -> List[Optional[int]]:
        return [None]

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: PTTargetPoint) -> str:
        return nncf_graph.get_node_by_name(target_point.target_node_name).layer_name

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names

    @staticmethod
    def get_weight_config(config: QuantizerConfig, model: NNCFNetwork) -> QuantizerConfig:
        return config

    @staticmethod
    def _get_input_scale_shape(
        nncf_graph: NNCFGraph, target_point: PTTargetPoint, quantization_config: QuantizerConfig
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
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

        scale_shape = tuple(
            get_scale_shape(
                input_shape, is_weights=is_weights, per_channel=quantization_config.per_channel, channel_idx=channel_idx
            )
        )

        return input_shape, scale_shape, channel_idx

    @staticmethod
    def _default_collector_params_and_scale_shape(
        nncf_graph: NNCFGraph, target_point: PTTargetPoint, quantizer_config: QuantizerConfig
    ) -> Tuple[PTRangeInitCollectorParams, Tuple[int, ...]]:
        input_shape, scale_shape, channel_idx = PTMinMaxAlgoBackend._get_input_scale_shape(
            nncf_graph, target_point, quantizer_config
        )
        return (
            PTRangeInitCollectorParams(
                is_weights=target_point.is_weight_target_point(),
                mode=quantizer_config.mode,
                per_channel=quantizer_config.per_channel,
                input_shape=input_shape,
                channel_idx=channel_idx,
            ),
            scale_shape,
        )

    @staticmethod
    def _statistic_collector_builder(
        collector_name: str,
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        num_samples: int = None,
    ) -> PTMeanMinMaxStatisticCollector:
        collector_params, scale_shape = PTMinMaxAlgoBackend._default_collector_params_and_scale_shape(
            nncf_graph, target_point, quantizer_config
        )
        init_config = RangeInitConfig(collector_name, num_samples)
        return StatCollectorGenerator.generate_stat_collector_for_range_init_config(
            init_config, scale_shape, collector_params, num_samples
        )

    @staticmethod
    def _create_quantizer(
        quantizer_config: QuantizerConfig,
        scale_shape: Tuple,
        parameters: FakeQuantizeParameters,
        target_type: TargetType,
    ) -> BaseQuantizer:
        mode = quantizer_config.mode
        quantizer_cls = QUANTIZATION_MODULES.get(mode)
        narrow_range = target_type == TargetType.OPERATION_WITH_WEIGHTS and mode == QuantizationMode.SYMMETRIC
        quantizer_spec = PTQuantizerSpec.from_config(
            quantizer_config,
            narrow_range=narrow_range,
            scale_shape=scale_shape,
            half_range=False,
            logarithm_scale=False,
            is_quantized_on_export=False,
            compression_lr_multiplier=None,
        )
        quantizer = quantizer_cls(quantizer_spec)

        # Fill it with minmax
        PTMinMaxAlgoBackend._fill_quantizer_parameters(quantizer, parameters)
        return quantizer

    @staticmethod
    def _fill_quantizer_parameters(quantizer: BaseQuantizer, parameters: FakeQuantizeParameters) -> None:
        quantizer.eps = 0
        if isinstance(quantizer, AsymmetricQuantizer):
            quantizer.input_low = torch.nn.Parameter(torch.from_numpy(parameters.input_low))
            quantizer.input_range = torch.nn.Parameter(
                torch.from_numpy(np.array(parameters.input_high - parameters.input_low))
            )
        else:
            quantizer.signed = np.any(parameters.input_low < 0)
            quantizer.scale = torch.nn.Parameter(torch.from_numpy(parameters.input_high))

    @staticmethod
    def _create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> PTInsertionCommand:
        _, scale_shape, _ = PTMinMaxAlgoBackend._get_input_scale_shape(nncf_graph, target_point, quantizer_config)

        quantizer = PTMinMaxAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, target_point.target_type
        )

        return PTInsertionCommand(target_point, quantizer, TransformationPriority.QUANTIZATION_PRIORITY)

    @staticmethod
    def get_model_type_ignore_scope(model_type: ModelType, device: TargetDevice) -> IgnoredScope:
        if model_type == ModelType.TRANSFORMER:
            types = []
            metatypes_to_add = [om.PTAddMetatype, om.PTPowerMetatype, om.PTSubMetatype, om.PTMeanMetatype]
            if device != TargetDevice.CPU_SPR:
                metatypes_to_add.append(om.PTMulMetatype)
            type_name_to_add = ["squeeze"]
            for metatype in metatypes_to_add:
                types.extend(metatype.get_all_aliases())
            types.extend(type_name_to_add)
            return IgnoredScope(types=types)
        return IgnoredScope()

    @staticmethod
    def get_weight_nodes(nncf_graph: NNCFGraph) -> List[NNCFNode]:
        return [
            node for node in nncf_graph.get_all_nodes() if isinstance(node.layer_attributes, WeightedLayerAttributes)
        ]
