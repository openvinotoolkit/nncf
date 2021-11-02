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
    get_per_channel_history, expand_like, percentile_reduce_like
from nncf.torch.tensor_statistics.statistics import PTMinMaxTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTMedianMADTensorStatistic
from nncf.torch.tensor_statistics.statistics import PTPercentileTensorStatistic


class PTOfflineTensorStatisticCollector(OfflineTensorStatisticCollector):
    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            self._samples.append(x.detach().cpu().numpy())


class MinMaxStatisticCollector(OnlineTensorStatisticCollector):
    def __init__(self, reduction_shapes: Set[ReductionShape] = None,
                 rs_vs_params: Dict[ReductionShape, Dict[str, Union[str, bool]]] = None,
                 num_samples: int = None):
        super().__init__(reduction_shapes, num_samples)
        self._rs_vs_params = rs_vs_params
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
                if self._rs_vs_params is None:
                    mode = 'symmetric'
                else:
                    mode = self._rs_vs_params[reduction_shape].get('mode', 'symmetric')
                if mode == 'symmetric':
                    max_reduced = max_reduce_like(torch.abs(x), reduction_shape)
                else:
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

    def _get_statistics(self) -> Dict[ReductionShape, PTMinMaxTensorStatistic]:
        return {rs: PTMinMaxTensorStatistic(self._min_values[rs], self._max_values[rs])
                for rs in self._reduction_shapes}

    def _reset(self):
        self._min_values = {rs: None for rs in self._reduction_shapes}  # type: Dict[ReductionShape]
        self._max_values = {rs: None for rs in self._reduction_shapes}  # type: Dict[ReductionShape]


class MixedMinMaxStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self,
                 is_weights: bool,
                 reduction_shapes: Set[ReductionShape] = None,
                 rs_vs_params: Dict[ReductionShape, Dict[str, Union[str, bool]]] = None,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shapes, num_samples, window_size)
        self._is_weights = is_weights
        self._window_size = window_size
        self._all_min_values = {}  # type: Dict[ReductionShape, Deque]
        self._all_max_values = {}  # type: Dict[ReductionShape, Deque]
        if not rs_vs_params:
            self._rs_vs_params = {}
        else:
            self._rs_vs_params = rs_vs_params
        self._mode = {}  # type: Dict[ReductionShape, str]
        self._per_channel = {}  # type: Dict[ReductionShape, bool]
        if self._reduction_shapes is not None:
            for rs in self._reduction_shapes:
                self._all_min_values[rs] = deque(maxlen=window_size)
                self._all_max_values[rs] = deque(maxlen=window_size)
                self._mode[rs] = self._rs_vs_params.get(rs, {'mode': 'symmetric'}).get('mode', 'symmetric')
                self._per_channel[rs] = self._rs_vs_params.get(rs, {'per_channel': False}).get('per_channel', False)

    def _reduction_shape_per_sample(self, x, rs):
        # Collect statistics per sample for activations
        if not self._is_weights:
            if self._per_channel[rs]:
                rs = (x.shape[0],) + rs[1:]
            else:
                rs = (x.shape[0],) + (1,) * (x.dim() - 1)
        return rs

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for reduction_shape in self._reduction_shapes:
                if reduction_shape not in self._all_min_values:
                    self._all_min_values[reduction_shape] = deque(maxlen=self._window_size)
                if reduction_shape not in self._all_max_values:
                    self._all_max_values[reduction_shape] = deque(maxlen=self._window_size)
                if reduction_shape not in self._mode:
                    self._mode[reduction_shape] = \
                        self._rs_vs_params.get(reduction_shape, {'mode': 'symmetric'}).get('mode', 'symmetric')
                if reduction_shape not in self._per_channel:
                    self._per_channel[reduction_shape] = \
                        self._rs_vs_params.get(reduction_shape, {'per_channel': False}).get('per_channel', False)

                reduction_shape_per_sample = self._reduction_shape_per_sample(x, reduction_shape)
                min_reduced = min_reduce_like(x, reduction_shape_per_sample)
                if self._mode[reduction_shape] == 'symmetric':
                    max_reduced = max_reduce_like(torch.abs(x), reduction_shape)
                else:
                    max_reduced = max_reduce_like(x, reduction_shape)
                if self._is_weights:
                    self._all_min_values[reduction_shape].append(min_reduced)
                    self._all_max_values[reduction_shape].append(max_reduced)
                else:
                    self._all_min_values[reduction_shape].extend([t.view(reduction_shape)
                                                                  for t in torch.unbind(min_reduced)])
                    self._all_max_values[reduction_shape].extend([t.view(reduction_shape)
                                                                  for t in torch.unbind(max_reduced)])

    def _reset(self):
        for rs in self._reduction_shapes:
            self._all_min_values[rs].clear()
            self._all_max_values[rs].clear()

    def _get_statistics(self) -> Dict[ReductionShape, PTMinMaxTensorStatistic]:
        retval = {}
        for rs in self._reduction_shapes:
            stacked_min = torch.stack(list(self._all_min_values[rs]))
            if not self._is_weights and not self._per_channel[rs] and self._mode[rs] == 'asymmetric':
                min_values = stacked_min.mean(dim=0)
            else:
                min_values, _ = stacked_min.min(dim=0)

            stacked_max = torch.stack(list(self._all_max_values[rs]))
            if not self._is_weights and not self._per_channel[rs]:
                max_values = stacked_max.mean(dim=0)
            else:
                max_values, _ = stacked_max.max(dim=0)
            retval[rs] = PTMinMaxTensorStatistic(min_values.view(rs), max_values.view(rs))
        return retval


class MeanMinMaxStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self, reduction_shapes: Set[ReductionShape] = None,
                 rs_vs_params: Dict[ReductionShape, Dict[str, Union[str, bool]]] = None,
                 num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shapes, num_samples, window_size)
        self._window_size = window_size
        self._all_min_values = {}  # type: Dict[ReductionShape, Deque]
        self._all_max_values = {}  # type: Dict[ReductionShape, Deque]
        if not rs_vs_params:
            self._rs_vs_params = {}
        else:
            self._rs_vs_params = rs_vs_params
        self._mode = {}  # type: Dict[ReductionShape, str]
        if self._reduction_shapes is not None:
            for rs in self._reduction_shapes:
                self._all_min_values[rs] = deque(maxlen=window_size)
                self._all_max_values[rs] = deque(maxlen=window_size)
                self._mode[rs] = self._rs_vs_params.get(rs, {'mode': 'symmetric'}).get('mode', 'symmetric')

    def _register_input(self, x: torch.Tensor):
        with no_nncf_trace():
            for reduction_shape in self._reduction_shapes:
                if reduction_shape not in self._all_min_values:
                    self._all_min_values[reduction_shape] = deque(maxlen=self._window_size)
                if reduction_shape not in self._all_max_values:
                    self._all_max_values[reduction_shape] = deque(maxlen=self._window_size)
                self._all_min_values[reduction_shape].append(min_reduce_like(x, reduction_shape))
                if self._mode[reduction_shape] == 'symmetric':
                    self._all_max_values[reduction_shape].append(max_reduce_like(torch.abs(x), reduction_shape))
                else:
                    self._all_max_values[reduction_shape].append(max_reduce_like(x, reduction_shape))

    def _reset(self):
        for rs in self._reduction_shapes:
            self._all_min_values[rs].clear()
            self._all_max_values[rs].clear()

    def _get_statistics(self) -> Dict[ReductionShape, PTMinMaxTensorStatistic]:
        retval = {}
        for rs in self._reduction_shapes:
            stacked_min = torch.stack(list(self._all_min_values[rs]))
            min_values = stacked_min.mean(dim=0).view(rs)

            stacked_max = torch.stack(list(self._all_max_values[rs]))
            max_values = stacked_max.mean(dim=0).view(rs)
            retval[rs] = PTMinMaxTensorStatistic(min_values, max_values)
        return retval


class MedianMADStatisticCollector(PTOfflineTensorStatisticCollector):
    def _get_statistics(self) -> Dict[ReductionShape, PTMedianMADTensorStatistic]:
        retval = {} # type: Dict[ReductionShape, PTMedianMADTensorStatistic]
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
            retval[reduction_shape] = PTMedianMADTensorStatistic(median_tensor, mad_tensor)

        return retval


class PercentileStatisticCollector(PTOfflineTensorStatisticCollector):
    def __init__(self, percentiles_to_collect: List[float],
                 reduction_shapes: Set[ReductionShape] = None,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shapes, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect  # NB: Percentiles are valued between 0 and 100

    def _get_statistics(self) -> Dict[ReductionShape, PTPercentileTensorStatistic]:
        retval = {}  # type: Dict[ReductionShape, PTPercentileTensorStatistic]
        for reduction_shape in self._reduction_shapes:
            per_channel_history = get_per_channel_history(self._samples, reduction_shape)
            percentile_vs_values_dict = {}  # type: Dict[float, torch.Tensor]
            for pc in self._percentiles_to_collect:
                per_channel_percentiles = [np.percentile(channel_hist, pc) for channel_hist in per_channel_history]
                numpy_percentiles = np.asarray(per_channel_percentiles)
                torch_percentiles = torch.from_numpy(numpy_percentiles).to(dtype=torch.float)
                torch_percentiles = expand_like(torch_percentiles, reduction_shape)
                percentile_vs_values_dict[pc] = torch_percentiles
            retval[reduction_shape] = PTPercentileTensorStatistic(percentile_vs_values_dict)
        return retval


class MeanPercentileStatisticCollector(PTOfflineTensorStatisticCollector):
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
            for pct, val in self._all_pct_values.items():
                for reduction_shape in self._reduction_shapes:
                    if reduction_shape not in val:
                        val[reduction_shape] = deque(maxlen=self._window_size)
                    val[reduction_shape].append(percentile_reduce_like(x, reduction_shape, pct))

    def _reset(self):
        for _, val in self._all_pct_values.items():
            for rs in self._reduction_shapes:
                val[rs].clear()

    def _get_statistics(self) -> Dict[ReductionShape, PTPercentileTensorStatistic]:
        retval = {}
        for rs in self._reduction_shapes:
            mean_percentile_values = {}
            for pct, val in self._all_pct_values.items():
                stacked_pct_vals = torch.stack(list(val[rs]))
                mean_percentile_values[pct] = stacked_pct_vals.mean(dim=0).view(rs)
            retval[rs] = PTPercentileTensorStatistic(mean_percentile_values)
        return retval
