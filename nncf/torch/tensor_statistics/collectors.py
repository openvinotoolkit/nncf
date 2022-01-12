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

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
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
from nncf.torch.tensor import PTNNCFTensor


class PTNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for PTNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return PTNNCFTensor(torch.amin(x.tensor, dim=axis))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return PTNNCFTensor(torch.amax(x.tensor, dim=axis))

    @staticmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.abs(x.tensor))

    @staticmethod
    def min(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.min(x1.tensor, x2.tensor))

    @staticmethod
    def max(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return PTNNCFTensor(torch.max(x1.tensor, x2.tensor))

    @staticmethod
    def mean(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return PTNNCFTensor(x.tensor.mean(dim=axis))

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return PTNNCFTensor(torch.stack(x, dim=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        tensor_list = torch.unbind(x.tensor, dim=axis)
        return [PTNNCFTensor(t) for t in tensor_list]

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return torch.sum(tensor.tensor).item()


class PTMinMaxStatisticCollector(MinMaxStatisticCollector):
    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape, output_shape: ReductionShape,
                 num_samples: int = None):
        super().__init__(use_abs_max, reduction_shape, num_samples)
        self._output_shape = output_shape

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return PTNNCFCollectorTensorProcessor()

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

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return PTNNCFCollectorTensorProcessor()

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
        super().__init__(use_per_sample_stats, use_abs_max, reduction_shape,
                                                             num_samples, window_size)
        self._output_shape = output_shape

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return PTNNCFCollectorTensorProcessor()

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._register_input_common(PTNNCFTensor(x))

    def _get_statistics(self) -> PTMinMaxTensorStatistic:
        min_values = self._min_aggregate().tensor.view(self._output_shape)
        max_values = self._max_aggregate().tensor.view(self._output_shape)
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
