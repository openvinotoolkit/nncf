# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

import nncf
from nncf.common.graph.layer_attributes import WeightedLayerAttributes
from nncf.common.quantization.initialization.range import RangeInitCollectorParams
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.common.quantization.quantizer_setup import QuantizationPointBase
from nncf.common.quantization.quantizer_setup import QuantizerSetupBase
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.scopes import should_consider_scope
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.config.schemata.algo.quantization import RANGE_INIT_TYPES_VS_DESCRIPTIONS
from nncf.experimental.common.tensor_statistics.collectors import AggregationAxes
from nncf.torch.graph.graph import PTNNCFGraph
from nncf.torch.initialization import DataLoaderBaseRunner
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import SymmetricQuantizer
from nncf.torch.quantization.layers import get_scale_shape
from nncf.torch.quantization.translator import PTTargetPointTranslator
from nncf.torch.tensor_statistics.algo import TensorStatisticObservationPoint
from nncf.torch.tensor_statistics.algo import create_register_input_hook
from nncf.torch.tensor_statistics.collectors import get_mean_percentile_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_median_mad_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_min_max_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_mixed_min_max_statistic_collector
from nncf.torch.tensor_statistics.collectors import get_percentile_tensor_collector
from nncf.torch.tensor_statistics.statistics import pt_convert_stat_to_min_max_tensor_stat


class PTRangeInitParams(RangeInitParams):
    def get_max_num_init_steps(self) -> int:
        steps = []
        if self.global_init_config is not None:
            steps.append(self.global_init_config.num_init_samples)
        for pl_config in self.per_layer_range_init_configs:
            steps.append(pl_config.num_init_samples)
        batch_size = self.init_range_data_loader.batch_size
        return int(np.ceil(max(steps) / batch_size))

    def get_init_config_for_quantization_point(self, qp: QuantizationPointBase) -> RangeInitConfig:
        if qp.is_weight_quantization_point():
            qid = WeightQuantizerId(qp.insertion_point.target_node_name)
            group = QuantizerGroup.WEIGHTS
        else:
            qid = NonWeightQuantizerId(qp.insertion_point.target_node_name, qp.insertion_point.input_port_id)
            group = QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(qid, group)

    def get_init_config_for_scope_and_group(self, qid: QuantizerId, group: QuantizerGroup) -> RangeInitConfig:
        matches: List[RangeInitConfig] = []
        for pl_config in self.per_layer_range_init_configs:
            should_be_considered = should_consider_scope(qid, pl_config.ignored_scopes, pl_config.target_scopes)
            if should_be_considered and (group == pl_config.target_group or pl_config.target_group is None):
                matches.append(
                    RangeInitConfig(
                        pl_config.init_type, pl_config.num_init_samples, pl_config.init_type_specific_params
                    )
                )
        if len(matches) > 1:
            raise ValueError(
                "Location {} matches more than one per-layer initialization parameter definition!".format(str(qid))
            )
        if len(matches) == 1:
            return matches[0]
        if not matches and self.global_init_config is not None:
            return deepcopy(self.global_init_config)

        raise ValueError(
            "Location {} does not match any per-layer initialization parameter definition!".format(str(qid))
        )


class PTRangeInitCollectorParams(RangeInitCollectorParams):
    def __init__(
        self, is_weights: bool, scheme: QuantizationScheme, per_channel: bool, input_shape: tuple, channel_idx: int
    ):
        """
        :param is_weights: Boolean that defines tensor type. True for Weights, False for Activations.
        :param scheme: Quantization scheme: symmetric or asymmetric.
        :param input_shape: Shape of the input tensor.
        :param channel_idx: Channel dimension.
        """
        super().__init__(is_weights, scheme, per_channel)
        self._input_shape = input_shape
        self._channel_idx = channel_idx

    def get_reduction_aggregation_axes(self, is_per_sample: bool) -> Tuple[ReductionAxes, AggregationAxes]:
        if self.is_per_channel:
            return super().get_reduction_aggregation_axes(self._input_shape, (self._channel_idx,), is_per_sample)
        return super().get_reduction_aggregation_axes(self._input_shape, (), is_per_sample)


class StatCollectorGenerator:
    @staticmethod
    def generate_collectors_for_range_init_statistics_collection(
        target_model_graph: PTNNCFGraph, quantizer_setup: QuantizerSetupBase, range_init_params: PTRangeInitParams
    ) -> Dict[TensorStatisticObservationPoint, Dict[ReductionAxes, TensorStatisticCollectorBase]]:
        retval = {}
        for qp in quantizer_setup.quantization_points.values():
            init_config = range_init_params.get_init_config_for_quantization_point(qp)
            is_weights = qp.is_weight_quantization_point()
            num_batches = int(
                np.ceil(init_config.num_init_samples / range_init_params.init_range_data_loader.batch_size)
            )
            if is_weights:
                # No need to store extra statistics in memory since weights won't change during range init
                num_batches = 1

            tp = PTTargetPointTranslator.translate(qp.insertion_point)
            scale_shapes_vs_params = StatCollectorGenerator.get_all_scale_shapes_with_params(qp, target_model_graph)

            obs_p = TensorStatisticObservationPoint(tp, reduction_shapes=set(scale_shapes_vs_params.keys()))

            retval[obs_p] = {}
            for scale_shape in obs_p.reduction_shapes:
                collector_params = scale_shapes_vs_params[scale_shape]
                collector = StatCollectorGenerator.generate_stat_collector_for_range_init_config(
                    init_config, scale_shape, collector_params, num_samples_to_collect_override=num_batches
                )
                retval[obs_p][scale_shape] = collector

        return retval

    @staticmethod
    def generate_stat_collector_for_range_init_config(
        init_config: RangeInitConfig,
        scale_shape: ReductionAxes = None,
        collector_params: PTRangeInitCollectorParams = None,
        num_samples_to_collect_override: int = None,
    ) -> TensorStatisticCollectorBase:
        num_samples = init_config.num_init_samples
        if num_samples_to_collect_override is not None:
            num_samples = num_samples_to_collect_override
        if init_config.init_type not in RANGE_INIT_TYPES_VS_DESCRIPTIONS:
            raise nncf.InternalError("Unknown range init type: {}".format(init_config.init_type))

        use_per_sample_stats = collector_params.use_per_sample_stats(init_config.init_type == "mixed_min_max")
        reduction_axes, aggregation_axes = collector_params.get_reduction_aggregation_axes(use_per_sample_stats)
        if init_config.init_type == "min_max":
            return get_min_max_statistic_collector(
                use_abs_max=collector_params.use_abs_max,
                reduction_axes=reduction_axes,
                aggregation_axes=aggregation_axes,
                scale_shape=scale_shape,
                num_samples=num_samples,
            )
        if init_config.init_type == "mixed_min_max":
            return get_mixed_min_max_statistic_collector(
                use_abs_max=collector_params.use_abs_max,
                reduction_axes=reduction_axes,
                aggregation_axes=aggregation_axes,
                scale_shape=scale_shape,
                use_means_of_mins=collector_params.use_means_of_mins,
                use_means_of_maxs=collector_params.use_means_of_maxs,
                num_samples=num_samples,
            )
        if init_config.init_type == "mean_min_max":
            return get_mixed_min_max_statistic_collector(
                use_abs_max=collector_params.use_abs_max,
                reduction_axes=reduction_axes,
                aggregation_axes=aggregation_axes,
                scale_shape=scale_shape,
                use_means_of_mins=True,
                use_means_of_maxs=True,
                num_samples=num_samples,
            )
        if init_config.init_type == "threesigma":
            return get_median_mad_statistic_collector(
                reduction_axes=reduction_axes,
                aggregation_axes=aggregation_axes,
                scale_shape=scale_shape,
                num_samples=num_samples,
            )
        if init_config.init_type == "percentile":
            min_percentile = init_config.init_type_specific_params.get("min_percentile", 0.1)
            max_percentile = init_config.init_type_specific_params.get("max_percentile", 99.9)
            return get_percentile_tensor_collector(
                percentiles_to_collect=(min_percentile, max_percentile),
                reduction_axes=reduction_axes,
                aggregation_axes=aggregation_axes,
                scale_shape=scale_shape,
                num_samples=num_samples,
            )

        if init_config.init_type == "mean_percentile":
            min_percentile = init_config.init_type_specific_params.get("min_percentile", 0.1)
            max_percentile = init_config.init_type_specific_params.get("max_percentile", 99.9)
            return get_mean_percentile_statistic_collector(
                percentiles_to_collect=(min_percentile, max_percentile),
                reduction_axes=reduction_axes,
                aggregation_axes=aggregation_axes,
                scale_shape=scale_shape,
                num_samples=num_samples,
            )
        raise ValueError("Range init type not handled!")

    @classmethod
    def get_all_scale_shapes_with_params(
        cls, qp: QuantizationPointBase, target_nncf_graph: PTNNCFGraph
    ) -> Dict[ReductionAxes, PTRangeInitCollectorParams]:
        qconfigs = qp.get_all_configs_list()
        if qp.is_weight_quantization_point():
            module_node = target_nncf_graph.get_node_by_name(qp.insertion_point.target_node_name)
            layer_attributes = module_node.layer_attributes
            assert isinstance(layer_attributes, WeightedLayerAttributes)
            input_shape = layer_attributes.get_weight_shape()
            channel_idx = layer_attributes.get_target_dim_for_compression()
        else:
            input_shape = target_nncf_graph.get_input_shape_for_insertion_point(qp.insertion_point)
            channel_idx = 1  # channel dim for activations

        retval = {}
        for qconfig in qconfigs:
            is_weights = qp.is_weight_quantization_point()
            scale_shape = tuple(
                get_scale_shape(
                    input_shape, is_weights=is_weights, per_channel=qconfig.per_channel, channel_idx=channel_idx
                )
            )

            if scale_shape not in retval:
                retval[scale_shape] = PTRangeInitCollectorParams(
                    is_weights, qconfig.mode, qconfig.per_channel, input_shape, channel_idx
                )
        return retval


class DataLoaderRangeInitializeRunner(DataLoaderBaseRunner):
    def __init__(
        self,
        model: NNCFNetwork,
        modules_to_init_vs_init_configs: Dict[str, Tuple[BaseQuantizer, RangeInitConfig, bool, Tuple[int]]],
        init_device: str,
        batch_size: int = None,
    ):
        super().__init__(model, init_device)
        self.modules_to_init = modules_to_init_vs_init_configs
        self.progressbar_description = "Range parameters initialization"

        self.collectors_and_modules_to_init: Dict[str, Tuple[TensorStatisticCollectorBase, BaseQuantizer]] = (
            OrderedDict()
        )
        self.hook_handles = []
        self.batch_size = batch_size

    def _get_fwd_hook(
        self, collector: TensorStatisticCollectorBase
    ) -> Callable[["torch.Module", torch.Tensor, torch.Tensor], torch.Tensor]:
        hook = create_register_input_hook(collector=collector)

        def fwd_hook(module, input_, output):
            hook(input_[0])

        return fwd_hook

    def _prepare_initialization(self):
        for name, data in self.modules_to_init.items():
            quantizer_module, init_config, is_weights, input_shape = data
            num_samples_override = None
            if self.batch_size is not None:
                num_batches = np.ceil(init_config.num_init_samples / self.batch_size)
                num_samples_override = num_batches

            if isinstance(quantizer_module, SymmetricQuantizer):
                mode = QuantizationScheme.SYMMETRIC
            else:
                mode = QuantizationScheme.ASYMMETRIC

            shape = quantizer_module.scale_shape
            if shape == (1,):  # Per-tensor
                channel_idx = None
            elif len(shape) > 1 and all(item == 1 for item in shape):
                channel_idx = 0  # (1, 1, 1, 1) - doest not matter which dim is channel_idx
            else:
                if not is_weights:
                    channel_idx = 1  # channel dim for activations
                else:
                    channel_idx = [i for i, val in enumerate(shape) if val != 1][0]

            collector_params = PTRangeInitCollectorParams(
                is_weights, mode, quantizer_module.per_channel, input_shape, channel_idx
            )

            collector = StatCollectorGenerator.generate_stat_collector_for_range_init_config(
                init_config, tuple(quantizer_module.scale_shape), collector_params, num_samples_override
            )

            self.collectors_and_modules_to_init[name] = collector, quantizer_module

            self.hook_handles.append(quantizer_module.register_forward_hook(self._get_fwd_hook(collector)))

    def _apply_initializers(self):
        for handle in self.hook_handles:
            handle.remove()
        for scope_str, collector_and_module in self.collectors_and_modules_to_init.items():
            collector, quantizer_module = collector_and_module
            target_stat = collector.get_statistics()
            minmax_stats = pt_convert_stat_to_min_max_tensor_stat(target_stat)
            quantizer_module.apply_minmax_init(
                minmax_stats.min_values.data, minmax_stats.max_values.data, log_module_name=scope_str
            )
