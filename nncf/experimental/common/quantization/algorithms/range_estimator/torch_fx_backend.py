# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Set, Tuple

import torch
from torch.quantization.fake_quantize import FakeQuantize

import nncf
from nncf.common.graph.graph import NNCFGraph
from nncf.common.graph.graph import NNCFNode
from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.common.quantization.algorithms.range_estimator.backend import RangeEstimatorAlgoBackend
from nncf.experimental.common.tensor_statistics.collectors import AGGREGATORS_MAP
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.torch.fx.commands import FXApplyTransformationCommand
from nncf.experimental.torch.fx.model_utils import get_target_point
from nncf.experimental.torch.fx.transformations import qdq_insertion_transformation_builder
from nncf.quantization.advanced_parameters import StatisticsType
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.graph.graph import PTTargetPoint
from nncf.torch.graph.transformations.commands import PTSharedFnInsertionCommand
from nncf.torch.model_graph_manager import get_weight_tensor_port_ids
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import get_scale_shape
from nncf.torch.quantization.strip import convert_to_torch_fakequantizer
from nncf.torch.tensor_statistics.collectors import PT_REDUCERS_MAP


class FXRangeEstimatorAlgoBackend(RangeEstimatorAlgoBackend):
    @staticmethod
    def target_point(target_type: TargetType, target_node_name: str, port_id: int) -> PTTargetPoint:
        return get_target_point(target_type, target_node_name, port_id)

    @staticmethod
    def get_target_point_shape(nncf_graph: PTNNCFGraph, node: NNCFNode, target_point: PTTargetPoint) -> Tuple[int, ...]:
        return nncf_graph.get_input_shape_for_insertion_point(target_point)

    @staticmethod
    def get_weight_quantization_axes(node: NNCFNode, target_point: PTTargetPoint, ndims: int) -> Tuple[int]:
        # TODO(dlyakhov): support transpose conv and other cases
        return (0,)

    @staticmethod
    def get_statistic_collector(
        range_estimator_params: RangeEstimatorParameters,
        use_abs_max: bool,
        reduction_axes: Optional[Tuple[int, ...]],
        aggregation_axes: Optional[Tuple[int, ...]],
        inplace: bool,
        num_samples: Optional[int] = None,
    ) -> TensorCollector:
        collector = TensorCollector(MinMaxTensorStatistic)
        for params, container_key in zip(
            [range_estimator_params.min, range_estimator_params.max],
            [MinMaxTensorStatistic.MIN_STAT, MinMaxTensorStatistic.MAX_STAT],
        ):
            if params.statistics_type not in PT_REDUCERS_MAP:
                raise nncf.InternalError(
                    f"Statistic type: {params.statistics_type} is not supported for Torch PTQ backend yet."
                )

            if params.aggregator_type not in AGGREGATORS_MAP:
                raise nncf.InternalError(
                    f"Aggregator type: {params.aggregator_type} is not supported for Torch PTQ backend yet."
                )

            statistic_type = params.statistics_type
            if statistic_type in [StatisticsType.QUANTILE, StatisticsType.ABS_QUANTILE]:
                # TODO(dlyakhov): merge two quantile aggregators in one
                if container_key == MinMaxTensorStatistic.MIN_STAT:
                    quantile = params.quantile_outlier_prob
                else:
                    quantile = 1 - params.quantile_outlier_prob
                reducer = PT_REDUCERS_MAP[statistic_type](reduction_axes=reduction_axes, quantile=[quantile])
            else:
                if use_abs_max and statistic_type == StatisticsType.MAX:
                    statistic_type = StatisticsType.ABS_MAX
                reducer = PT_REDUCERS_MAP[statistic_type](reduction_axes=reduction_axes)

            kwargs = {
                "num_samples": num_samples,
                "aggregation_axes": aggregation_axes,
            }
            if params.aggregator_type in [AggregatorType.MEAN_NO_OUTLIERS, AggregatorType.MEDIAN_NO_OUTLIERS]:
                kwargs.update({"quantile": params.quantile_outlier_prob})
            aggregator = AGGREGATORS_MAP[params.aggregator_type](**kwargs)

            collector.register_statistic_branch(container_key, reducer, aggregator)
        return collector

    @staticmethod
    def get_weight_tensor_port_ids(node: NNCFNode, graph: NNCFGraph) -> List[Optional[int]]:
        return get_weight_tensor_port_ids(node, graph)

    @staticmethod
    def get_weight_name(nncf_graph: NNCFGraph, target_point: PTTargetPoint) -> str:
        weighted_node = nncf_graph.get_node_by_name(target_point.target_node_name)
        weight_edge = nncf_graph.get_input_edge_by_port_id(weighted_node, target_point.input_port_id)
        weight = weight_edge.from_node
        return weight.node_name

    @staticmethod
    def should_quantize_weight(weight_name: str, quantized_weight_names: Set[str]) -> bool:
        # If the nodes share one weight tensor, we should have only one quantizer on that
        return weight_name not in quantized_weight_names

    @staticmethod
    def _get_input_scale_shape(
        nncf_graph: NNCFGraph, target_point: PTTargetPoint, per_channel: bool
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], int]:
        is_weights = target_point.is_weight_target_point()
        if is_weights:
            # TODO(dlyakhov): support transpose conv/ make channel_idx common
            channel_idx = 0
        else:
            channel_idx = 1  # channel dim for activations

        input_shape = nncf_graph.get_input_shape_for_insertion_point(target_point)
        scale_shape = tuple(
            get_scale_shape(input_shape, is_weights=is_weights, per_channel=per_channel, channel_idx=channel_idx)
        )

        return input_shape, scale_shape, channel_idx

    @staticmethod
    def _create_quantizer(
        quantizer_config: QuantizerConfig,
        scale_shape: Tuple,
        parameters: FakeQuantizeParameters,
        target_type: TargetType,
    ) -> FakeQuantize:
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
        # TODO(dlyakhov) Prevent creation of intermediate objects like nncf quantizer.
        FXRangeEstimatorAlgoBackend._fill_quantizer_parameters(quantizer, parameters, quantizer_spec.scale_shape)
        # Convert to the torch fake quantizer
        torch_fq = convert_to_torch_fakequantizer(quantizer)
        return torch_fq

    @staticmethod
    def _fill_quantizer_parameters(quantizer: BaseQuantizer, parameters: FakeQuantizeParameters, scale_shape) -> None:
        if isinstance(quantizer, AsymmetricQuantizer):
            quantizer.input_low = torch.nn.Parameter(parameters.input_low.data.reshape(scale_shape))
            input_range = parameters.input_high - parameters.input_low
            # Subtract eps from the input_range to make quantizer parameters equal to
            # original parameters on the forward call.
            quantizer.input_range = torch.nn.Parameter((input_range.data - quantizer.eps).reshape(scale_shape))
        else:
            quantizer.signed = bool(torch.any(parameters.input_low.data < 0))
            # Subtract eps from the scale to make quantizer parameters equal to
            # original parameters on the forward call.
            quantizer.scale = torch.nn.Parameter((parameters.input_high.data - quantizer.eps).reshape(scale_shape))

    @staticmethod
    def create_quantizer_insertion_command(
        nncf_graph: NNCFGraph,
        target_point: PTTargetPoint,
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> FXApplyTransformationCommand:
        _, scale_shape, _ = FXRangeEstimatorAlgoBackend._get_input_scale_shape(
            nncf_graph, target_point, quantizer_config.per_channel
        )

        quantizer = FXRangeEstimatorAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, target_point.target_type
        )
        transformation = qdq_insertion_transformation_builder(quantizer, [target_point])
        return FXApplyTransformationCommand(transformation)

    @staticmethod
    def create_unified_scales_quantizers_insertion_commands(
        nncf_graph: NNCFGraph,
        target_points: List[PTTargetPoint],
        quantizer_config: QuantizerConfig,
        parameters: FakeQuantizeParameters,
    ) -> List[PTSharedFnInsertionCommand]:
        _, scale_shape, _ = FXRangeEstimatorAlgoBackend._get_input_scale_shape(
            nncf_graph, target_points[0], quantizer_config.per_channel
        )

        quantizer = FXRangeEstimatorAlgoBackend._create_quantizer(
            quantizer_config, scale_shape, parameters, target_points[0].target_type
        )

        transformations = []
        for tp in target_points:
            transformation = qdq_insertion_transformation_builder(quantizer, [tp])
            transformations.append(FXApplyTransformationCommand(transformation))
        return transformations
