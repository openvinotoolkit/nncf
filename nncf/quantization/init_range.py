import queue
from copy import deepcopy
from typing import List, Dict, Set, Optional

import numpy as np
import torch

from nncf.dynamic_graph.context import no_nncf_trace
from nncf.nncf_logger import logger as nncf_logger
from nncf.quantization.layers import BaseQuantizer
from nncf.quantization.quantizer_id import WeightQuantizerId, NonWeightQuantizerId
from nncf.quantization.quantizer_setup import QuantizationPointBase, QuantizerSetupBase
from nncf.quantization.structs import QuantizerGroup
from nncf.tensor_statistics.algo import TensorStatisticObservationPoint
from nncf.tensor_statistics.collectors import TensorStatisticCollectorBase, MinMaxStatisticCollector, ReductionShape, \
    MeanMinMaxStatisticCollector, MedianMADStatisticCollector, PercentileStatisticCollector
from nncf.utils import get_flat_tensor_contents_string, should_consider_scope


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

    def generate_stat_collector(self, reduction_shapes: Set[ReductionShape],
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
        matches = []  # type: List[RangeInitConfig]
        for pl_config in self.per_layer_range_init_configs:
            if should_consider_scope(scope_str, pl_config.target_scopes, pl_config.ignored_scopes):
                if group == pl_config.target_group or pl_config.target_group is None:
                    matches.append(RangeInitConfig(pl_config.init_type, pl_config.num_init_samples,
                                                   pl_config.init_type_specific_params))
        if len(matches) > 1:
            raise ValueError("Location {} matches more than one per-layer initialization parameter definition in NNCF"
                             " config file!".format(qp.insertion_point))
        if len(matches) == 1:
            return matches[0]
        if not matches and self.global_init_config is not None:
            return deepcopy(self.global_init_config)

        raise ValueError("Location {} does not match any per-layer initialization parameter definition in NNCF"
                         " config file!".format(qp.insertion_point))


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


class MinMaxInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer', num_init_steps: int, log_module_name: str = None):
        super().__init__(quantize_module, num_init_steps)
        self.min_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        self.max_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)
        self.log_module_name = log_module_name

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.min_values = torch.min(min_reduce_like(x, self.scale_shape),
                                        self.min_values)
            self.max_values = torch.max(max_reduce_like(x, self.scale_shape),
                                        self.max_values)

    def reset(self):
        self.min_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        self.max_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)

    def apply_init(self):
        nncf_logger.debug("Statistics: min={} max={}".format(get_flat_tensor_contents_string(self.min_values),
                                                             get_flat_tensor_contents_string(self.max_values)))
        self.quantize_module.apply_minmax_init(self.min_values, self.max_values, self.log_module_name)


class MeanMinMaxInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer', num_init_steps: int, log_module_name: str = None):
        super().__init__(quantize_module, num_init_steps)
        self.log_module_name = log_module_name
        self.all_min_values = []
        self.all_max_values = []

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.all_min_values.append(min_reduce_like(x, self.scale_shape))
            self.all_max_values.append(max_reduce_like(x, self.scale_shape))

    def reset(self):
        self.all_min_values.clear()
        self.all_max_values.clear()

    def apply_init(self):
        min_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        max_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)
        if self.all_min_values:
            stacked_min = torch.stack(self.all_min_values)
            min_values = stacked_min.mean(dim=0).view(self.scale_shape)
        if self.all_max_values:
            stacked_max = torch.stack(self.all_max_values)
            max_values = stacked_max.mean(dim=0).view(self.scale_shape)
        nncf_logger.debug("Statistics: min={} max={}".format(get_flat_tensor_contents_string(min_values),
                                                             get_flat_tensor_contents_string(max_values)))
        self.quantize_module.apply_minmax_init(min_values, max_values, self.log_module_name)


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


class ThreeSigmaInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer', num_init_steps: int, log_module_name: str = None):
        super().__init__(quantize_module, num_init_steps)
        self.input_history = queue.Queue()
        self.log_module_name = log_module_name

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.input_history.put(x.detach().cpu().numpy())

    def reset(self):
        self.input_history = queue.Queue()

    def apply_init(self):
        self.medians = torch.ones(self.scale_shape).to(self.device)
        self.median_absolute_deviations = torch.ones(self.scale_shape).to(self.device)

        per_channel_history = get_per_channel_history(self.input_history, self.scale_shape,
                                                      discard_zeros=True)
        per_channel_median = [np.median(channel_hist) for channel_hist in per_channel_history]
        per_channel_mad = []
        for idx, median in enumerate(per_channel_median):
            # Constant factor depends on the distribution form - assuming normal
            per_channel_mad.append(1.4826 * np.median(abs(per_channel_history[idx] - median)))

        numpy_median = np.asarray(per_channel_median)
        numpy_mad = np.asarray(per_channel_mad)
        median_tensor = torch.from_numpy(numpy_median).to(self.device, dtype=torch.float)
        mad_tensor = torch.from_numpy(numpy_mad).to(self.device, dtype=torch.float)

        median_tensor = expand_like(median_tensor, self.scale_shape)
        mad_tensor = expand_like(mad_tensor, self.scale_shape)

        nncf_logger.debug("Statistics: median={} MAD={}".format(get_flat_tensor_contents_string(median_tensor),
                                                                get_flat_tensor_contents_string(mad_tensor)))
        self.quantize_module.apply_minmax_init(median_tensor - 3 * mad_tensor, median_tensor + 3 * mad_tensor,
                                               self.log_module_name)


class PercentileInitializer(QuantizeRangeInitializer):
    def __init__(self, quantize_module: 'BaseQuantizer',
                 num_init_steps: int,
                 min_percentile: float,
                 max_percentile: float,
                 log_module_name: str = None):
        super().__init__(quantize_module, num_init_steps)
        self.input_history = queue.Queue()
        self.log_module_name = log_module_name
        self.min_percentile = min_percentile  # NB: Both min_percentile and max_percentile
        self.max_percentile = max_percentile  # are valued between 0 and 100

    def register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self.input_history.put(x.detach().cpu().numpy())

    def reset(self):
        self.input_history = queue.Queue()

    def apply_init(self):
        self.min_values = torch.ones(self.scale_shape).to(self.device) * np.inf
        self.max_values = torch.ones(self.scale_shape).to(self.device) * (-np.inf)

        per_channel_history = get_per_channel_history(self.input_history, self.scale_shape)
        per_channel_min_percentiles = [np.percentile(channel_hist, self.min_percentile) for channel_hist in
                                       per_channel_history]
        per_channel_max_percentiles = [np.percentile(channel_hist, self.max_percentile) for channel_hist in
                                       per_channel_history]

        numpy_mins = np.asarray(per_channel_min_percentiles)
        numpy_maxs = np.asarray(per_channel_max_percentiles)
        mins_tensor = torch.from_numpy(numpy_mins).to(self.device, dtype=torch.float)
        maxs_tensor = torch.from_numpy(numpy_maxs).to(self.device, dtype=torch.float)

        mins_tensor = expand_like(mins_tensor, self.scale_shape)
        maxs_tensor = expand_like(maxs_tensor, self.scale_shape)

        nncf_logger.debug("Statistics: Min ({}%th) percentile = {},"
                          " Max ({}%th) percentile = {}".format(self.min_percentile,
                                                                get_flat_tensor_contents_string(mins_tensor),
                                                                self.max_percentile,
                                                                get_flat_tensor_contents_string(maxs_tensor)))
        self.quantize_module.apply_minmax_init(mins_tensor,
                                               maxs_tensor, self.log_module_name)


class RangeInitializerFactory:
    @staticmethod
    def create(init_config: Dict, module: torch.nn.Module, log_module_name: str):
        init_type = init_config["type"]
        num_init_samples = init_config["num_init_samples"]
        if init_type == "min_max":
            return MinMaxInitializer(module, num_init_samples, log_module_name)
        if init_type == "threesigma":
            return ThreeSigmaInitializer(module, num_init_samples, log_module_name)
        if init_type == "mean_min_max":
            return MeanMinMaxInitializer(module, num_init_samples, log_module_name)
        if init_type == "percentile":
            min_percentile = init_config.get("min_percentile", 10)
            max_percentile = init_config.get("max_percentile", 90)
            return PercentileInitializer(module, num_init_samples, min_percentile, max_percentile, log_module_name)
        raise NotImplementedError
