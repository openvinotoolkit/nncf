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
from typing import List, Dict, Deque

import torch
import numpy as np

from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MixedMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import OfflineTensorStatisticCollector
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.tensor_statistics.reduction import  get_per_channel_history, expand_like, percentile_reduce_like
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic
from nncf.torch.tensor import PTNNCFTensor


class PTMinMaxStatisticCollector(MinMaxStatisticCollector):
    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape, output_shape: ReductionShape,
                 num_samples: int = None):
        super().__init__(use_abs_max, reduction_shape, num_samples)
        self._output_shape = output_shape

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(PTNNCFTensor(x))

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_values.tensor.view(self._output_shape)
        max_values = self._max_values.tensor.view(self._output_shape)
        return PTMinMaxTensorStatistic(min_values, max_values)


class PTMixedMinMaxStatisticCollector(MixedMinMaxStatisticCollector):
    def __init__(self,
                 use_per_sample_stats: bool,
                 use_abs_max: bool,
                 use_means_of_mins: bool,
                 use_means_of_maxs: bool,
                 reduction_shape: ReductionShape,
                 output_shape: ReductionShape,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(use_per_sample_stats, use_abs_max, use_means_of_mins,
                         use_means_of_maxs, reduction_shape, num_samples, window_size)
        self._output_shape = output_shape

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(PTNNCFTensor(x))

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_aggregate().tensor.view(self._output_shape)
        max_values = self._max_aggregate().tensor.view(self._output_shape)
        return PTMinMaxTensorStatistic(min_values, max_values)


class PTMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    def __init__(self,
                 use_per_sample_stats: bool,
                 use_abs_max: bool,
                 reduction_shape: ReductionShape,
                 output_shape: ReductionShape,
                 num_samples: int = None,
                 window_size: int = None):
        super(PTMeanMinMaxStatisticCollector, self).__init__(use_per_sample_stats, use_abs_max, reduction_shape,
                                                             num_samples, window_size)
        self._output_shape = output_shape

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(PTNNCFTensor(x))

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_aggregate().tensor.view(self._output_shape)
        max_values = self._max_aggregate().tensor.view(self._output_shape)
        return PTMinMaxTensorStatistic(min_values, max_values)


class MedianMADStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates median and median absolute deviation (MAD).
    """

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())

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


class PercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values of all data history.
    """

    def __init__(self, percentiles_to_collect: List[float], reduction_shape: ReductionShape = None,
                 num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect  # NB: Percentiles are valued between 0 and 100

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())

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


class MeanPercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values per step and then averages the results.
    """

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
