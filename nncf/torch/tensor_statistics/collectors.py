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

from typing import Union, List, Deque

import torch

from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import CollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import MedianMADStatisticCollector
from nncf.common.tensor_statistics.collectors import PercentileStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanPercentileStatisticCollector
from nncf.common.tensor_statistics.collectors import MixedMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor_statistics.reduction import np_percentile_reduce_like
from nncf.torch.tensor_statistics.reduction import  expand_like
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic
from nncf.torch.dynamic_graph.context import no_nncf_trace


class PTCollectorTensorProcessor(CollectorTensorProcessor):
    """
    A PT realization of the processing methods set for PTNNCFTensors.
    """

    @classmethod
    def reduce_min(cls, x: torch.Tensor, axis: Union[int, tuple]) -> torch.Tensor:
        return torch.amin(x, dim=axis)

    @classmethod
    def reduce_max(cls, x: torch.Tensor, axis: Union[int, tuple]) -> torch.Tensor:
        return torch.amax(x, dim=axis)

    @classmethod
    def abs(cls, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    @classmethod
    def min(cls, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.min(x1, x2)

    @classmethod
    def max(cls, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return torch.max(x1, x2)

    @classmethod
    def mean(cls, x: torch.Tensor, axis: Union[int, tuple]) -> torch.Tensor:
        return x.mean(dim=axis)

    @classmethod
    def stack(cls, x: Union[List[torch.Tensor], Deque[torch.Tensor]], axis: int = 0) -> torch.Tensor:
        return torch.stack(tuple(x), dim=axis)

    @classmethod
    def unstack(cls, x: torch.Tensor, axis: int = 0) -> List[torch.Tensor]:
        return torch.unbind(x, dim=axis)


class PTMinMaxStatisticCollector(MinMaxStatisticCollector):
    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape, output_shape: ReductionShape,
                 num_samples: int = None):
        super().__init__(use_abs_max, reduction_shape, num_samples)
        self._output_shape = output_shape

    @staticmethod
    def _get_processor() -> CollectorTensorProcessor:
        return PTCollectorTensorProcessor()

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(x)

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_values.view(self._output_shape)
        max_values = self._max_values.view(self._output_shape)
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

    @staticmethod
    def _get_processor() -> CollectorTensorProcessor:
        return PTCollectorTensorProcessor()

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(x)

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_aggregate().view(self._output_shape)
        max_values = self._max_aggregate().view(self._output_shape)
        return PTMinMaxTensorStatistic(min_values, max_values)


class PTMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    def __init__(self,
                 use_per_sample_stats: bool,
                 use_abs_max: bool,
                 reduction_shape: ReductionShape,
                 output_shape: ReductionShape,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(use_per_sample_stats, use_abs_max, reduction_shape,
                                                             num_samples, window_size)
        self._output_shape = output_shape

    @staticmethod
    def _get_processor() -> CollectorTensorProcessor:
        return PTCollectorTensorProcessor()

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(x)

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_aggregate().view(self._output_shape)
        max_values = self._max_aggregate().view(self._output_shape)
        return PTMinMaxTensorStatistic(min_values, max_values)


class PTMedianMADStatisticCollector(MedianMADStatisticCollector):
    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())

    def _get_statistics(self) -> PTMedianMADTensorStatistic:
        numpy_median, numpy_mad = self._prepare_statistics()
        median_tensor = torch.from_numpy(numpy_median).to(dtype=torch.float)
        mad_tensor = torch.from_numpy(numpy_mad).to(dtype=torch.float)

        median_tensor = expand_like(median_tensor, list(self._reduction_shape))
        mad_tensor = expand_like(mad_tensor, list(self._reduction_shape))

        return PTMedianMADTensorStatistic(median_tensor, mad_tensor)


class PTPercentileStatisticCollector(PercentileStatisticCollector):
    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())

    def _get_statistics(self) -> PTPercentileTensorStatistic:
        percentile_vs_values_dict = self._prepare_statistics()
        for key, val in percentile_vs_values_dict.items():
            torch_percentiles = torch.from_numpy(val).to(dtype=torch.float)
            percentile_vs_values_dict[key] = expand_like(torch_percentiles, list(self._reduction_shape))
        return PTPercentileTensorStatistic(percentile_vs_values_dict)


class PTMeanPercentileStatisticCollector(MeanPercentileStatisticCollector):
    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for pct, val in self._all_pct_values.items():
                np_vals = np_percentile_reduce_like(x.cpu().numpy(), self._reduction_shape, pct)
                torch_vals = torch.from_numpy(np_vals).to(dtype=torch.float)
                val.append(torch_vals)

    def _get_statistics(self) -> PTPercentileTensorStatistic:
        mean_percentile_values = {}
        for pct, val in self._all_pct_values.items():
            stacked_pct_vals = torch.stack(list(val))
            mean_percentile_values[pct] = stacked_pct_vals.mean(dim=0).view(self._reduction_shape)
        return PTPercentileTensorStatistic(mean_percentile_values)
