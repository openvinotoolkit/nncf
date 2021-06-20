"""
 Copyright (c) 2021 Intel Corporation
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

from collections import OrderedDict
from copy import deepcopy
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import torch

from nncf.common.graph import NNCFGraph
from nncf.common.quantization.initialization.range import RangeInitConfig
from nncf.common.quantization.initialization.range import RangeInitParams
from nncf.common.quantization.structs import NonWeightQuantizerId
from nncf.common.quantization.structs import QuantizerGroup
from nncf.common.quantization.structs import QuantizerId
from nncf.common.quantization.structs import WeightQuantizerId
from nncf.common.utils.helpers import should_consider_scope
from nncf.torch.initialization import DataLoaderBaseRunner
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.quantizer_setup import QuantizationPointBase
from nncf.torch.quantization.quantizer_setup import QuantizerSetupBase
from nncf.torch.tensor_statistics.algo import TensorStatisticObservationPoint
from nncf.torch.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.torch.tensor_statistics.collectors import MeanPercentileStatisticCollector
from nncf.torch.tensor_statistics.collectors import MedianMADStatisticCollector
from nncf.torch.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.torch.tensor_statistics.collectors import PercentileStatisticCollector
from nncf.torch.tensor_statistics.collectors import ReductionShape
from nncf.torch.tensor_statistics.collectors import TensorStatisticCollectorBase
from nncf.torch.tensor_statistics.statistics import MinMaxTensorStatistic


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
            qid = NonWeightQuantizerId(qp.insertion_point.target_node_name,
                                       qp.insertion_point.input_port_id)
            group = QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(qid, group)

    def get_init_config_for_scope_and_group(self, qid: QuantizerId, group: QuantizerGroup) -> RangeInitConfig:
        matches = []  # type: List[RangeInitConfig]
        for pl_config in self.per_layer_range_init_configs:
            if should_consider_scope(qid, pl_config.ignored_scopes, pl_config.target_scopes):
                if group == pl_config.target_group or pl_config.target_group is None:
                    matches.append(RangeInitConfig(pl_config.init_type, pl_config.num_init_samples,
                                                   pl_config.init_type_specific_params))
        if len(matches) > 1:
            raise ValueError("Location {} matches more than one per-layer initialization parameter "
                             "definition!".format(str(qid)))
        if len(matches) == 1:
            return matches[0]
        if not matches and self.global_init_config is not None:
            return deepcopy(self.global_init_config)

        raise ValueError("Location {} does not match any per-layer initialization parameter "
                         "definition!".format(str(qid)))


class StatCollectorGenerator:
    @staticmethod
    def generate_collectors_for_range_init_statistics_collection(target_model_graph: NNCFGraph,
                                                                 quantizer_setup: QuantizerSetupBase,
                                                                 range_init_params: PTRangeInitParams) -> \
            Dict[TensorStatisticObservationPoint, TensorStatisticCollectorBase]:
        retval = {}
        for qp in quantizer_setup.quantization_points.values():
            init_config = range_init_params.get_init_config_for_quantization_point(qp)
            is_weights = qp.is_weight_quantization_point()
            num_batches = int(np.ceil(
                init_config.num_init_samples / range_init_params.init_range_data_loader.batch_size))
            if is_weights:
                module_node = target_model_graph.get_node_by_name(qp.insertion_point.target_node_name)
                input_shape = module_node.layer_attributes.get_weight_shape()
                # No need to store extra statistics in memory since weights won't change during range init
                num_batches = 1
            else:
                input_shape = target_model_graph.get_input_shape_for_insertion_point(qp.insertion_point)

            obs_p = TensorStatisticObservationPoint(
                qp.insertion_point,
                reduction_shapes=set(qp.get_all_scale_shapes(input_shape)))

            collector = StatCollectorGenerator.generate_stat_collector_for_range_init_config(
                init_config,
                obs_p.reduction_shapes,
                num_samples_to_collect_override=num_batches)
            retval[obs_p] = collector
        return retval

    @staticmethod
    def generate_stat_collector_for_range_init_config(
            init_config: RangeInitConfig,
            reduction_shapes: Set[ReductionShape] = None,
            num_samples_to_collect_override: int = None) -> TensorStatisticCollectorBase:
        num_samples = init_config.num_init_samples
        if num_samples_to_collect_override is not None:
            num_samples = num_samples_to_collect_override
        if init_config.init_type == "min_max":
            return MinMaxStatisticCollector(reduction_shapes, num_samples)
        if init_config.init_type == "mean_min_max":
            return MeanMinMaxStatisticCollector(reduction_shapes, num_samples)
        if init_config.init_type == "threesigma":
            return MedianMADStatisticCollector(reduction_shapes, num_samples)
        if init_config.init_type == "percentile":
            min_percentile = init_config.init_type_specific_params.get("min_percentile", 0.1)
            max_percentile = init_config.init_type_specific_params.get("max_percentile", 99.9)
            return PercentileStatisticCollector([min_percentile, max_percentile], reduction_shapes, num_samples)
        if init_config.init_type == "mean_percentile":
            min_percentile = init_config.init_type_specific_params.get("min_percentile", 0.1)
            max_percentile = init_config.init_type_specific_params.get("max_percentile", 99.9)
            return MeanPercentileStatisticCollector([min_percentile, max_percentile], reduction_shapes, num_samples)
        raise RuntimeError("Unknown range init type: {}".format(init_config.init_type))


class DataLoaderRangeInitializeRunner(DataLoaderBaseRunner):
    def __init__(
            self,
            model: NNCFNetwork,
            modules_to_init_vs_init_configs: Dict[str, Tuple[torch.nn.Module, RangeInitConfig]],
            init_device: str,
            batch_size: int = None
    ):
        super().__init__(model, init_device)
        self.modules_to_init = modules_to_init_vs_init_configs
        self.progressbar_description = 'Range parameters initialization'

        #pylint:disable=line-too-long
        self.collectors_and_modules_to_init = OrderedDict()  # type: Dict[str, Tuple[TensorStatisticCollectorBase, BaseQuantizer]]
        self.hook_handles = []
        self.batch_size = batch_size

    def _get_fwd_hook(self, collector: TensorStatisticCollectorBase) -> Callable:
        def fwd_hook(module, input_, output):
            collector.register_input(input_[0])
        return fwd_hook

    def _prepare_initialization(self):
        for name, data in self.modules_to_init.items():
            quantizer_module, init_config = data  # type: BaseQuantizer, RangeInitConfig
            num_samples_override = None
            if self.batch_size is not None:
                num_batches = np.ceil(init_config.num_init_samples / self.batch_size)
                num_samples_override = num_batches

            collector = StatCollectorGenerator.generate_stat_collector_for_range_init_config(
                init_config,
                {tuple(quantizer_module.scale_shape)},
                num_samples_override
            )

            self.collectors_and_modules_to_init[name] = collector, quantizer_module

            self.hook_handles.append(
                quantizer_module.register_forward_hook(self._get_fwd_hook(collector))
            )

    def _apply_initializers(self):
        for handle in self.hook_handles:
            handle.remove()
        for scope_str, collector_and_module in self.collectors_and_modules_to_init.items():
            collector, quantizer_module = collector_and_module
            scale_shape = tuple(quantizer_module.scale_shape)
            target_stat = collector.get_statistics()[scale_shape]
            minmax_stats = MinMaxTensorStatistic.from_stat(target_stat)
            quantizer_module.apply_minmax_init(minmax_stats.min_values, minmax_stats.max_values,
                                               log_module_name=scope_str)
