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

from collections import deque
from typing import List, Dict, Set, Deque, Union

import torch
import numpy as np

from nncf.common.tensor_statistics.collectors import OnlineTensorStatisticCollector
from nncf.common.tensor_statistics.collectors import OfflineTensorStatisticCollector
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.tensor_statistics.reduction import min_reduce_like, max_reduce_like, \
    get_per_channel_history, expand_like, percentile_reduce_like, get_reduction_shapes
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic


class PTOfflineTensorStatisticCollector(OfflineTensorStatisticCollector):
    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())


class MinMaxStatisticCollector(OnlineTensorStatisticCollector):
    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape = None, num_samples: int = None):
        super().__init__(reduction_shape, num_samples)
        self._use_abs_max = use_abs_max

        self._min_values = None
        self._max_values = None

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            min_reduced = min_reduce_like(x, list(self._reduction_shape))
            if self._use_abs_max:
                max_reduced = max_reduce_like(torch.abs(x), list(self._reduction_shape))
            else:
                max_reduced = max_reduce_like(x, list(self._reduction_shape))

            if self._min_values is None:
                self._min_values = min_reduced
            else:
                self._min_values = torch.min(min_reduced, self._min_values)

            if self._max_values is None:
                self._max_values = max_reduced
            else:
                self._max_values = torch.max(max_reduced, self._max_values)

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        return PTMinMaxTensorStatistic(self._min_values, self._max_values)

    def _reset(self):
        self._min_values = None
        self._max_values = None


class MixedMinMaxStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self, use_per_sample_stats: bool, use_abs_max: bool, use_means_of_mins: bool, use_means_of_maxs: bool,
                 reduction_shape: bool = None, num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples)
        self._per_channel = reduction_shape != (1,)
        self._use_per_sample_stats = use_per_sample_stats
        self._use_abs_max = use_abs_max
        self._use_means_of_mins = use_means_of_mins
        self._use_means_of_maxs = use_means_of_maxs

        self._all_min_values = deque(maxlen=window_size)
        self._all_max_values = deque(maxlen=window_size)

    def _reduction_shape_per_sample(self, x, rs):
        """Updates reduction shape if statistics are collected per-sample"""
        if self._use_per_sample_stats:
            if self._per_channel:
                rs = (x.shape[0],) + rs[1:]
            else:
                rs = (x.shape[0],) + (1,) * (x.dim() - 1)
        return rs

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            reduction_shape = self._reduction_shape_per_sample(x, self._reduction_shape)
            min_reduced = min_reduce_like(x, reduction_shape)
            if self._use_abs_max:
                max_reduced = max_reduce_like(torch.abs(x), reduction_shape)
            else:
                max_reduced = max_reduce_like(x, reduction_shape)

            if self._use_per_sample_stats:
                self._all_min_values.extend([t.view(self._reduction_shape)
                                                              for t in torch.unbind(min_reduced)])
                self._all_max_values.extend([t.view(self._reduction_shape)
                                                              for t in torch.unbind(max_reduced)])
            else:
                self._all_min_values.append(min_reduced)
                self._all_max_values.append(max_reduced)

    def _reset(self):
        self._all_min_values.clear()
        self._all_max_values.clear()

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        stacked_min = torch.stack(list(self._all_min_values))
        if self._use_means_of_mins:
            min_values = stacked_min.mean(dim=0)
        else:
            min_values, _ = stacked_min.min(dim=0)

        stacked_max = torch.stack(list(self._all_max_values))
        if self._use_means_of_maxs:
            max_values = stacked_max.mean(dim=0)
        else:
            max_values, _ = stacked_max.max(dim=0)
        return PTMinMaxTensorStatistic(min_values.view(self._reduction_shape), max_values.view(self._reduction_shape))


class MeanMinMaxStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape = None,
                 num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples)
        self._use_abs_max = use_abs_max
        self._all_min_values = deque(maxlen=window_size)
        self._all_max_values = deque(maxlen=window_size)

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._all_min_values.append(min_reduce_like(x, list(self._reduction_shape)))
            if self._use_abs_max:
                self._all_max_values.append(max_reduce_like(torch.abs(x), list(self._reduction_shape)))
            else:
                self._all_max_values.append(max_reduce_like(x, list(self._reduction_shape)))

    def _reset(self):
        self._all_min_values.clear()
        self._all_max_values.clear()

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        stacked_min = torch.stack(list(self._all_min_values))
        min_values = stacked_min.mean(dim=0).view(self._reduction_shape)

        stacked_max = torch.stack(list(self._all_max_values))
        max_values = stacked_max.mean(dim=0).view(self._reduction_shape)
        return PTMinMaxTensorStatistic(min_values, max_values)


class MedianMADStatisticCollector(PTOfflineTensorStatisticCollector):
    def _get_statistics(self) -> PTMedianMADTensorStatistic:
        per_channel_history = get_per_channel_history(self._samples, list(self._reduction_shape),
                                                      discard_zeros=True)
        per_channel_median = [np.median(channel_hist) for channel_hist in per_channel_history]
        per_channel_mad = []
        for idx, median in enumerate(per_channel_median):
            per_channel_mad.append(np.median(abs(per_channel_history[idx] - median)))

        numpy_median = np.asarray(per_channel_median)
        numpy_mad = np.asarray(per_channel_mad)
        median_tensor = torch.from_numpy(numpy_median).to(dtype=torch.float)
        mad_tensor = torch.from_numpy(numpy_mad).to(dtype=torch.float)

        median_tensor = expand_like(median_tensor, list(self._reduction_shape))
        mad_tensor = expand_like(mad_tensor, list(self._reduction_shape))

        return PTMedianMADTensorStatistic(median_tensor, mad_tensor)


class PercentileStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self, percentiles_to_collect: List[float], reduction_shape: ReductionShape = None,
                 num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect  # NB: Percentiles are valued between 0 and 100

    def _get_statistics(self) -> PTPercentileTensorStatistic:
        per_channel_history = get_per_channel_history(self._samples, list(self._reduction_shape))
        percentile_vs_values_dict = {}  # type: Dict[float, torch.Tensor]
        for pc in self._percentiles_to_collect:
            per_channel_percentiles = [np.percentile(channel_hist, pc) for channel_hist in per_channel_history]
            numpy_percentiles = np.asarray(per_channel_percentiles)
            torch_percentiles = torch.from_numpy(numpy_percentiles).to(dtype=torch.float)
            torch_percentiles = expand_like(torch_percentiles, list(self._reduction_shape))
            percentile_vs_values_dict[pc] = torch_percentiles
        return PTPercentileTensorStatistic(percentile_vs_values_dict)


class MeanPercentileStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self, percentiles_to_collect: List[float], reduction_shape: ReductionShape = None,
                 num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._all_pct_values = {}  # type: Dict[float, Deque]
        for pc in percentiles_to_collect:
            self._all_pct_values[pc] = deque(maxlen=window_size)

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for pct, val in self._all_pct_values.items():
                val.append(percentile_reduce_like(x, list(self._reduction_shape), pct))

    def _reset(self):
        for _, val in self._all_pct_values.items():
            val.clear()

    def _get_statistics(self) -> PTPercentileTensorStatistic:
        mean_percentile_values = {}
        for pct, val in self._all_pct_values.items():
            stacked_pct_vals = torch.stack(list(val))
            mean_percentile_values[pct] = stacked_pct_vals.mean(dim=0).view(self._reduction_shape)
        return PTPercentileTensorStatistic(mean_percentile_values)
