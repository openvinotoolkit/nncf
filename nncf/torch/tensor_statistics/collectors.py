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

from abc import ABC, abstractmethod
from collections import deque
from typing import List, Dict, Set, Tuple, Deque

import torch
import numpy as np

from nncf.torch.dynamic_graph.context import no_nncf_trace
from nncf.torch.tensor_statistics.reduction import min_reduce_like, max_reduce_like, \
    get_per_channel_history, expand_like, percentile_reduce_like
from nncf.torch.tensor_statistics.statistics import TensorStatistic, MinMaxTensorStatistic, \
    MedianMADTensorStatistic, PercentileTensorStatistic

ReductionShape = Tuple[int]


class TensorStatisticCollectorBase(ABC):
    def __init__(self, reduction_shapes: Set[ReductionShape] = None, num_samples: int = None):
        self._reduction_shapes = reduction_shapes
        self._enabled = True
        self._collected_samples = 0
        self._num_samples = num_samples

    def register_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self._enabled or \
                self._num_samples is not None and self._collected_samples >= self._num_samples:
            return x
        if self._reduction_shapes is None:
            self._reduction_shapes = {x.shape}
        self._register_input(x)
        self._collected_samples += 1
        return x

    @abstractmethod
    def _register_input(self, x: torch.Tensor):
        pass

    def get_statistics(self) -> Dict[ReductionShape, TensorStatistic]:
        if self._collected_samples == 0:
            raise StatisticsNotCollectedError()
        return self._get_statistics()

    @abstractmethod
    def _get_statistics(self) -> Dict[ReductionShape, TensorStatistic]:
        pass

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset(self):
        self._collected_samples = 0
        self._reset()

    @abstractmethod
    def _reset(self):
        pass

    def collected_samples(self) -> int:
        return self._collected_samples


class StatisticsNotCollectedError(Exception):
    pass


class OnlineTensorStatisticCollector(TensorStatisticCollectorBase, ABC):
    pass


class OfflineTensorStatisticCollector(TensorStatisticCollectorBase, ABC):
    def __init__(self, reduction_shapes: Set[ReductionShape] = None, num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shapes, num_samples)
        self._samples = deque(maxlen=window_size)  # type: Deque[torch.Tensor]

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())

    def _reset(self):
        self._samples.clear()


class MinMaxStatisticCollector(OnlineTensorStatisticCollector):
    def __init__(self, reduction_shapes: Set[ReductionShape] = None, num_samples: int = None):
        super().__init__(reduction_shapes, num_samples)
        if self._reduction_shapes is not None:
            self._min_values = {rs: None for rs in self._reduction_shapes}  # type: Dict[ReductionShape]
            self._max_values = {rs: None for rs in self._reduction_shapes}  # type: Dict[ReductionShape]
        else:
            self._min_values = {}  # type: Dict[ReductionShape]
            self._max_values = {}  # type: Dict[ReductionShape]

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for reduction_shape in self._reduction_shapes:
                min_reduced = min_reduce_like(x, reduction_shape)
                max_reduced = max_reduce_like(x, reduction_shape)
                # Have to use .get() because the inferred reduction shape is only known at first register_input call
                if self._min_values.get(reduction_shape) is None:
                    self._min_values[reduction_shape] = min_reduced
                else:
                    self._min_values[reduction_shape] = torch.min(min_reduced, self._min_values[reduction_shape])

                if self._max_values.get(reduction_shape) is None:
                    self._max_values[reduction_shape] = max_reduced
                else:
                    self._max_values[reduction_shape] = torch.max(max_reduced, self._max_values[reduction_shape])

    def _get_statistics(self) -> Dict[ReductionShape, MinMaxTensorStatistic]:
        return {rs: MinMaxTensorStatistic(self._min_values[rs], self._max_values[rs]) for rs in self._reduction_shapes}

    def _reset(self):
        self._min_values = {rs: None for rs in self._reduction_shapes}  # type: Dict[ReductionShape]
        self._max_values = {rs: None for rs in self._reduction_shapes}  # type: Dict[ReductionShape]


class MeanMinMaxStatisticCollector(OfflineTensorStatisticCollector):
    def __init__(self, reduction_shapes: Set[ReductionShape] = None, num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shapes, num_samples, window_size)
        self._window_size = window_size
        self._all_min_values = {}  # type: Dict[ReductionShape, Deque]
        self._all_max_values = {}  # type: Dict[ReductionShape, Deque]
        if self._reduction_shapes is not None:
            for rs in self._reduction_shapes:
                self._all_min_values[rs] = deque(maxlen=window_size)
                self._all_max_values[rs] = deque(maxlen=window_size)

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for reduction_shape in self._reduction_shapes:
                if reduction_shape not in self._all_min_values:
                    self._all_min_values[reduction_shape] = deque(maxlen=self._window_size)
                if reduction_shape not in self._all_max_values:
                    self._all_max_values[reduction_shape] = deque(maxlen=self._window_size)
                self._all_min_values[reduction_shape].append(min_reduce_like(x, reduction_shape))
                self._all_max_values[reduction_shape].append(max_reduce_like(x, reduction_shape))

    def _reset(self):
        for rs in self._reduction_shapes:
            self._all_min_values[rs].clear()
            self._all_max_values[rs].clear()

    def _get_statistics(self) -> Dict[ReductionShape, MinMaxTensorStatistic]:
        retval = {}
        for rs in self._reduction_shapes:
            stacked_min = torch.stack(list(self._all_min_values[rs]))
            min_values = stacked_min.mean(dim=0).view(rs)

            stacked_max = torch.stack(list(self._all_max_values[rs]))
            max_values = stacked_max.mean(dim=0).view(rs)
            retval[rs] = MinMaxTensorStatistic(min_values, max_values)
        return retval


class MedianMADStatisticCollector(OfflineTensorStatisticCollector):
    def _get_statistics(self) -> Dict[ReductionShape, MedianMADTensorStatistic]:
        retval = {} # type: Dict[ReductionShape, MedianMADTensorStatistic]
        for reduction_shape in self._reduction_shapes:
            per_channel_history = get_per_channel_history(self._samples, reduction_shape,
                                                          discard_zeros=True)
            per_channel_median = [np.median(channel_hist) for channel_hist in per_channel_history]
            per_channel_mad = []
            for idx, median in enumerate(per_channel_median):
                per_channel_mad.append(np.median(abs(per_channel_history[idx] - median)))

            numpy_median = np.asarray(per_channel_median)
            numpy_mad = np.asarray(per_channel_mad)
            median_tensor = torch.from_numpy(numpy_median).to(dtype=torch.float)
            mad_tensor = torch.from_numpy(numpy_mad).to(dtype=torch.float)

            median_tensor = expand_like(median_tensor, reduction_shape)
            mad_tensor = expand_like(mad_tensor, reduction_shape)
            retval[reduction_shape] = MedianMADTensorStatistic(median_tensor, mad_tensor)

        return retval


class PercentileStatisticCollector(OfflineTensorStatisticCollector):
    def __init__(self, percentiles_to_collect: List[float],
                 reduction_shapes: Set[ReductionShape] = None,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shapes, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect  # NB: Percentiles are valued between 0 and 100

    def _get_statistics(self) -> Dict[ReductionShape, PercentileTensorStatistic]:
        retval = {}  # type: Dict[ReductionShape, PercentileTensorStatistic]
        for reduction_shape in self._reduction_shapes:
            per_channel_history = get_per_channel_history(self._samples, reduction_shape)
            percentile_vs_values_dict = {}  # type: Dict[float, torch.Tensor]
            for pc in self._percentiles_to_collect:
                per_channel_percentiles = [np.percentile(channel_hist, pc) for channel_hist in per_channel_history]
                numpy_percentiles = np.asarray(per_channel_percentiles)
                torch_percentiles = torch.from_numpy(numpy_percentiles).to(dtype=torch.float)
                torch_percentiles = expand_like(torch_percentiles, reduction_shape)
                percentile_vs_values_dict[pc] = torch_percentiles
            retval[reduction_shape] = PercentileTensorStatistic(percentile_vs_values_dict)
        return retval


class MeanPercentileStatisticCollector(OfflineTensorStatisticCollector):
    def __init__(self, percentiles_to_collect: List[float],
                 reduction_shapes: Set[ReductionShape] = None,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shapes, num_samples, window_size)
        self._window_size = window_size
        self._all_pct_values = {}  # type: Dict[float, Dict[ReductionShape, Deque]]
        for pc in percentiles_to_collect:
            self._all_pct_values[pc] = {}
            if self._reduction_shapes is not None:
                for rs in self._reduction_shapes:
                    self._all_pct_values[pc][rs] = deque(maxlen=window_size)

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for pct in self._all_pct_values:
                for reduction_shape in self._reduction_shapes:
                    if reduction_shape not in self._all_pct_values[pct]:
                        self._all_pct_values[pct][reduction_shape] = deque(maxlen=self._window_size)
                    self._all_pct_values[pct][reduction_shape].append(percentile_reduce_like(x, reduction_shape, pct))

    def _reset(self):
        for pct in self._all_pct_values:
            for rs in self._reduction_shapes:
                self._all_pct_values[pct][rs].clear()

    def _get_statistics(self) -> Dict[ReductionShape, PercentileTensorStatistic]:
        retval = {}
        for rs in self._reduction_shapes:
            mean_percentile_values = {}
            for pct in self._all_pct_values:
                stacked_pct_vals = torch.stack(list(self._all_pct_values[pct][rs]))
                mean_percentile_values[pct] = stacked_pct_vals.mean(dim=0).view(rs)
            retval[rs] = PercentileTensorStatistic(mean_percentile_values)
        return retval
