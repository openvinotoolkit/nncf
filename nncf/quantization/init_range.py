import queue
from collections import OrderedDict
from copy import deepcopy
from typing import List, Dict, Set, Optional, Tuple, Callable

import numpy as np
import torch

from nncf.initialization import DataLoaderBaseRunner
from nncf.nncf_network import NNCFNetwork
from nncf.quantization.layers import BaseQuantizer
from nncf.quantization.quantizer_id import WeightQuantizerId, NonWeightQuantizerId
from nncf.quantization.quantizer_setup import QuantizationPointBase, QuantizerSetupBase
from nncf.quantization.structs import QuantizerGroup
from nncf.tensor_statistics.algo import TensorStatisticObservationPoint
from nncf.tensor_statistics.collectors import TensorStatisticCollectorBase, MinMaxStatisticCollector, ReductionShape, \
    MeanMinMaxStatisticCollector, MedianMADStatisticCollector, PercentileStatisticCollector
from nncf.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.utils import should_consider_scope


class RangeInitConfig:
    def __init__(self, init_type: str, num_init_samples: int, init_type_specific_params: Dict = None):
        self.init_type = init_type
        self.num_init_samples = num_init_samples
        self.init_type_specific_params = init_type_specific_params
        if self.init_type_specific_params is None:
            self.init_type_specific_params = {}

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    @classmethod
    def from_dict(cls, dct: Dict) -> 'RangeInitConfig':
        num_init_samples = dct.get("num_init_samples", 256)
        if num_init_samples < 0:
            raise ValueError("Number of initialization samples must be >= 0")
        return cls(dct.get("type", "mean_min_max"),
                   num_init_samples,
                   dct.get("params"))

    def generate_stat_collector(self, reduction_shapes: Set[ReductionShape] = None,
                                num_samples_to_collect_override: int = None) -> TensorStatisticCollectorBase:
        num_samples = self.num_init_samples
        if num_samples_to_collect_override is not None:
            num_samples = num_samples_to_collect_override
        if self.init_type == "min_max":
            return MinMaxStatisticCollector(reduction_shapes, num_samples)
        if self.init_type == "mean_min_max":
            return MeanMinMaxStatisticCollector(reduction_shapes, num_samples)
        if self.init_type == "threesigma":
            return MedianMADStatisticCollector(reduction_shapes, num_samples)
        if self.init_type == "percentile":
            min_percentile = self.init_type_specific_params.get("min_percentile", 10)
            max_percentile = self.init_type_specific_params.get("max_percentile", 90)
            return PercentileStatisticCollector([min_percentile, max_percentile], reduction_shapes, num_samples)
        raise RuntimeError("Unknown range init type: {}".format(self.init_type))


class PerLayerRangeInitConfig(RangeInitConfig):
    def __init__(self, range_init_config: RangeInitConfig,
                 target_scopes: Optional[List[str]],
                 ignored_scopes: Optional[List[str]],
                 target_quantizer_group: QuantizerGroup = None):
        super().__init__(range_init_config.init_type, range_init_config.num_init_samples,
                         range_init_config.init_type_specific_params)
        if target_scopes is None and ignored_scopes is None:
            raise ValueError("At least one of the (target_scopes, ignored_scopes) should be specified"
                             " for a per-layer range init config!")
        self.target_scopes = target_scopes
        self.ignored_scopes = ignored_scopes
        self.target_group = target_quantizer_group

    @classmethod
    def from_dict(cls, dct: Dict) -> 'PerLayerRangeInitConfig':
        base_config = RangeInitConfig.from_dict(dct)

        def get_list(dct: Dict, attr_name: str) -> Optional[List[str]]:
            str_or_list = dct.get(attr_name)
            if str_or_list is None:
                return None
            if isinstance(str_or_list, str):
                retval_list = [str_or_list]
            else:
                retval_list = str_or_list
            return retval_list
        target_scopes, ignored_scopes = get_list(dct, "target_scopes"), get_list(dct, "ignored_scopes")

        target_group_str = dct.get("target_quantizer_group")
        target_group = None
        if target_group_str is not None:
            target_group = QuantizerGroup.from_str(target_group_str)

        return cls(base_config, target_scopes, ignored_scopes, target_group)


class RangeInitParams:
    def __init__(self, init_range_data_loader: 'InitializingDataLoader',
                 device: str,
                 global_init_config: Optional[RangeInitConfig],
                 per_layer_range_init_configs: List[PerLayerRangeInitConfig]):
        self.init_range_data_loader = init_range_data_loader
        self.device = device
        self.global_init_config = global_init_config
        self.per_layer_range_init_configs = per_layer_range_init_configs

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
            scope_str = str(WeightQuantizerId(qp.insertion_point.module_scope))
            group = QuantizerGroup.WEIGHTS
        else:
            scope_str = str(NonWeightQuantizerId(qp.insertion_point.ia_op_exec_context,
                                                 qp.insertion_point.input_port_id))
            group = QuantizerGroup.ACTIVATIONS
        return self.get_init_config_for_scope_and_group(scope_str, group)

    def get_init_config_for_scope_and_group(self, scope_str: str, group: QuantizerGroup) -> RangeInitConfig:
        matches = []  # type: List[RangeInitConfig]
        for pl_config in self.per_layer_range_init_configs:
            if should_consider_scope(scope_str, pl_config.target_scopes, pl_config.ignored_scopes):
                if group == pl_config.target_group or pl_config.target_group is None:
                    matches.append(RangeInitConfig(pl_config.init_type, pl_config.num_init_samples,
                                                   pl_config.init_type_specific_params))
        if len(matches) > 1:
            raise ValueError("Location {} matches more than one per-layer initialization parameter "
                             "definition!".format(scope_str))
        if len(matches) == 1:
            return matches[0]
        if not matches and self.global_init_config is not None:
            return deepcopy(self.global_init_config)

        raise ValueError("Location {} does not match any per-layer initialization parameter "
                         "definition!".format(scope_str))


class StatCollectorGenerator:
    @staticmethod
    def generate_collectors_for_range_init_statistics_collection(quantizer_setup: QuantizerSetupBase,
                                                                 range_init_params: RangeInitParams) -> \
            Dict[TensorStatisticObservationPoint, TensorStatisticCollectorBase]:
        retval = {}
        for qp in quantizer_setup.quantization_points.values():
            obs_p = TensorStatisticObservationPoint(
                qp.insertion_point,
                reduction_shapes=set(qp.get_all_scale_shapes()))

            init_config = range_init_params.get_init_config_for_quantization_point(qp)
            is_weights = qp.is_weight_quantization_point()
            num_batches = int(np.ceil(
                init_config.num_init_samples / range_init_params.init_range_data_loader.batch_size))
            if is_weights:
                # No need to store extra statistics in memory since weights won't change during range init
                num_batches = 1

            collector = init_config.generate_stat_collector(obs_p.reduction_shapes,
                                                            num_samples_to_collect_override=num_batches)
            retval[obs_p] = collector
        return retval


class QuantizeRangeInitializer:
    def __init__(self, quantize_module: BaseQuantizer, num_init_steps: int):
        self.quantize_module = quantize_module
        self.device = next(self.quantize_module.parameters()).device
        self.scale_shape = self.quantize_module.scale_shape
        self.num_init_steps = num_init_steps
        self.num_register_init_steps = 0

    def register_input(self, x: torch.Tensor):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def forward_hook(self, module, input_, output):
        if self.num_register_init_steps < self.num_init_steps:
            self.register_input(input_[0])
            self.num_register_init_steps += 1

    def apply_init(self):
        raise NotImplementedError


def max_reduce_like(input_, ref_tensor_shape):
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        return input_.max()
    tmp_max = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            tmp_max, _ = torch.max(tmp_max, dim_idx, keepdim=True)
    return tmp_max


def min_reduce_like(input_, ref_tensor_shape):
    numel = np.prod(ref_tensor_shape)
    if numel == 1:
        return input_.min()
    tmp_min = input_
    for dim_idx, dim in enumerate(ref_tensor_shape):
        if dim == 1:
            tmp_min, _ = torch.min(tmp_min, dim_idx, keepdim=True)
    return tmp_min


def get_channel_count_and_dim_idx(scale_shape: List[int]):
    channel_dim_idx = 0
    channel_count = 1
    for dim_idx, dim in enumerate(scale_shape):
        if dim != 1:
            channel_dim_idx = dim_idx
            channel_count = dim
    return channel_count, channel_dim_idx


def expand_like(input_: torch.Tensor, scale_shape: List[int]):
    retval = input_
    count, idx = get_channel_count_and_dim_idx(scale_shape)
    assert input_.numel() == count
    assert len(input_.size()) == 1
    for _ in range(0, idx):
        retval = retval.unsqueeze(0)
    for _ in range(idx + 1, len(scale_shape)):
        retval = retval.unsqueeze(-1)
    return retval


def split_into_channels(input_: np.ndarray, scale_shape: List[int]) -> List[np.ndarray]:
    channel_count, channel_dim_idx = get_channel_count_and_dim_idx(scale_shape)
    channel_first_tensor = np.moveaxis(input_, channel_dim_idx, 0)
    if channel_count == 1:
        return [channel_first_tensor]

    ret_list = []
    for i in range(channel_count):
        ret_list.append(channel_first_tensor[i, ...])
    return ret_list


def get_per_channel_history(raw_input_history: queue.Queue, scale_shape: List[int], discard_zeros=False) -> List:
    channel_count, _ = get_channel_count_and_dim_idx(scale_shape)
    per_channel_history = [None for i in range(channel_count)]
    while not raw_input_history.empty():
        entry = raw_input_history.get()
        split = split_into_channels(entry, scale_shape)
        for i in range(channel_count):
            flat_channel_split = split[i].flatten()

            if discard_zeros:
                # For post-RELU quantizers exact zeros may prevail and lead to
                # zero mean and MAD - discard them
                flat_channel_split = flat_channel_split[flat_channel_split != 0]

            if per_channel_history[i] is None:
                per_channel_history[i] = flat_channel_split
            else:
                per_channel_history[i] = np.concatenate([per_channel_history[i], flat_channel_split])
    return per_channel_history


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

            collector = init_config.generate_stat_collector({tuple(quantizer_module.scale_shape)},
                                                            num_samples_override)

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
