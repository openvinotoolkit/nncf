# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC
from abc import abstractmethod
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

from nncf.common.tensor_statistics.reduction import ReductionAxes
from nncf.common.tensor_statistics.reduction import get_per_channel_history
from nncf.common.tensor_statistics.reduction import get_reduction_shape_from_sample_shape
from nncf.common.tensor_statistics.reduction import is_reduce_to_scalar
from nncf.common.tensor_statistics.reduction import percentile_reduce
from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.experimental.tensor import Tensor


class TensorStatisticCollectorBase(ABC):
    """Collector estimate statistics at the quantization point based on the provided reduction shape."""

    def __init__(self, reduction_axes: Optional[ReductionAxes] = None, num_samples: Optional[int] = None):
        """
        Initializes Tensor Statistic Collector

        :param reduction_axes: Shape that defines tensor dimensions to reduce.
        :param num_samples: Maximum number of samples to collect.
        """
        self._reduction_axes = reduction_axes
        self._enabled = True
        self._collected_samples = 0
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def register_input(self, x: Tensor) -> Tensor:
        """Registers input tensor"""
        if not self._enabled:
            return x
        if self._num_samples is not None and self._collected_samples >= self._num_samples:
            return x
        self._register_input(x)
        self._collected_samples += 1
        return x

    @abstractmethod
    def _register_input(self, x: Tensor):
        pass

    def get_statistics(self) -> TensorStatistic:
        """Returns collected statistics, if present."""
        if self._collected_samples == 0:
            raise StatisticsNotCollectedError()
        return self._get_statistics()

    @abstractmethod
    def _get_statistics(self) -> TensorStatistic:
        pass

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def reset(self):
        """Resets all the statistics in the collector."""
        self._collected_samples = 0
        self._reset()

    @abstractmethod
    def _reset(self):
        pass

    def collected_samples(self) -> int:
        return self._collected_samples

    def _get_axis(self) -> Optional[ReductionAxes]:
        return self._reduction_axes if not is_reduce_to_scalar(self._reduction_axes) else None


class StatisticsNotCollectedError(Exception):
    """Raised when the statistics are not collected but requested."""


class OnlineTensorStatisticCollector(TensorStatisticCollectorBase):
    """Base class for collectors that collects statistics in online regime, without storing the data."""


class OfflineTensorStatisticCollector(TensorStatisticCollectorBase):
    """Collects statistics in offline regime by storing the data and aggregating it afterwards."""

    def __init__(
        self, reduction_axes: Optional[ReductionAxes] = None, num_samples: int = None, window_size: int = None
    ):
        super().__init__(reduction_axes, num_samples)
        self._samples: Deque[Tensor] = deque(maxlen=window_size)

    def _reset(self):
        self._samples.clear()


class MinMaxStatisticCollector(OnlineTensorStatisticCollector):
    """Collector estimates min of minimum values and max of maximum values."""

    def __init__(self, use_abs_max: bool, reduction_axes: ReductionAxes, num_samples: int = None):
        super().__init__(reduction_axes, num_samples)
        self._use_abs_max = use_abs_max

        self._min_values = None
        self._max_values = None

    def _register_input(self, x: Tensor):
        from nncf.experimental.tensor import functions as fns

        axis = self._get_axis()
        min_reduced = fns.amin(x, axis=axis, keepdims=True)

        if self._use_abs_max:
            x = fns.abs(x)
        max_reduced = fns.amax(x, axis=axis, keepdims=True)

        if self._min_values is None:
            self._min_values = min_reduced
        else:
            self._min_values = fns.minimum(min_reduced, self._min_values)

        if self._max_values is None:
            self._max_values = max_reduced
        else:
            self._max_values = fns.maximum(max_reduced, self._max_values)

    def _reset(self):
        self._min_values = None
        self._max_values = None

    def _get_statistics(self) -> MinMaxTensorStatistic:
        return MinMaxTensorStatistic(self._min_values, self._max_values)


class MinMaxOfflineStatisticCollectorBase(OfflineTensorStatisticCollector):
    """
    Base class for collectors that aggregate statistics
    from minimum and maximum values of tensors.
    """

    def __init__(
        self,
        use_per_sample_stats: bool,
        use_abs_max: bool,
        reduction_axes: ReductionAxes,
        num_samples: int = None,
        window_size: int = None,
    ):
        super().__init__(reduction_axes, num_samples)
        self._use_per_sample_stats = use_per_sample_stats
        self._use_abs_max = use_abs_max

        self._all_min_values: Deque[Tensor] = deque(maxlen=window_size)
        self._all_max_values: Deque[Tensor] = deque(maxlen=window_size)

    def _register_input(self, x: Tensor):
        from nncf.experimental.tensor import functions as fns

        axis = self._get_axis()
        min_reduced = fns.amin(x, axis=axis, keepdims=True)
        if self._use_abs_max:
            x = fns.abs(x)
        max_reduced = fns.amax(x, axis=axis, keepdims=True)

        if self._use_per_sample_stats:
            self._all_min_values.extend(fns.unstack(min_reduced))
            self._all_max_values.extend(fns.unstack(max_reduced))
        else:
            self._all_min_values.append(min_reduced)
            self._all_max_values.append(max_reduced)

    @abstractmethod
    def _min_aggregate(self) -> Tensor:
        pass

    @abstractmethod
    def _max_aggregate(self) -> Tensor:
        pass

    def _reset(self):
        self._all_min_values.clear()
        self._all_max_values.clear()


class MixedMinMaxStatisticCollector(MinMaxOfflineStatisticCollectorBase):
    """
    Collector aggregates (min or mean) of minimum values and (max or mean) of maximum values.
    """

    def __init__(
        self,
        use_per_sample_stats: bool,
        use_abs_max: bool,
        use_means_of_mins: bool,
        use_means_of_maxs: bool,
        reduction_axes: ReductionAxes,
        num_samples: int = None,
        window_size: int = None,
    ):
        super().__init__(use_per_sample_stats, use_abs_max, reduction_axes, num_samples, window_size)
        self._use_means_of_mins = use_means_of_mins
        self._use_means_of_maxs = use_means_of_maxs

    def _min_aggregate(self) -> Tensor:
        from nncf.experimental.tensor import functions as fns

        stacked_min = fns.stack(list(self._all_min_values))
        if self._use_means_of_mins:
            return fns.mean(stacked_min, axis=0)
        return fns.amin(stacked_min, axis=0)

    def _max_aggregate(self) -> Tensor:
        from nncf.experimental.tensor import functions as fns

        stacked_max = fns.stack(list(self._all_max_values))
        if self._use_means_of_maxs:
            return fns.mean(stacked_max, axis=0)
        return fns.amin(stacked_max, axis=0)

    def _get_statistics(self) -> MinMaxTensorStatistic:
        return MinMaxTensorStatistic(self._min_aggregate(), self._max_aggregate())


class MeanMinMaxStatisticCollector(MinMaxOfflineStatisticCollectorBase):
    """
    Collector aggregates mean of minimum values and mean of maximum values.
    """

    def _min_aggregate(self) -> Tensor:
        from nncf.experimental.tensor import functions as fns

        stacked_min = fns.stack(list(self._all_min_values))
        return fns.mean(stacked_min, axis=0)

    def _max_aggregate(self) -> Tensor:
        from nncf.experimental.tensor import functions as fns

        stacked_max = fns.stack(list(self._all_max_values))
        return fns.mean(stacked_max, axis=0)

    def _get_statistics(self) -> MinMaxTensorStatistic:
        return MinMaxTensorStatistic(self._min_aggregate(), self._max_aggregate())


class MeanStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector that aggregates statistics as mean along a pre-assigned axis.
    """

    def __init__(
        self,
        reduction_axes: ReductionAxes = None,
        channel_axis: int = None,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ) -> None:
        """
        :param reduction_axes: The shape for the reduction while statistics collection.
        :param num_samples: Optional parameter for statistic collection that regulates
            the number of samples that will be processed.
        :param window_size: Optional maximum length for the statistic collection
        """
        if reduction_axes is None and channel_axis is None:
            raise RuntimeError("Either reduction_axes or channel_axis must be specified")

        if reduction_axes is not None and channel_axis is not None:
            raise RuntimeError("reduction_axes or channel_axis cannot be specified at the same time")

        super().__init__(reduction_axes, num_samples)
        self._channel_axis = channel_axis
        self._all_values: Deque[Tensor] = deque(maxlen=window_size)
        self._all_shapes: Deque[Tuple[int]] = deque(maxlen=window_size)

    def _register_input(self, x: Tensor):
        from nncf.experimental.tensor import functions as fns

        if self._reduction_axes is not None:
            axis = self._get_axis()
        else:  # self._channel_axis is not None
            proto_axis_list = list(range(x.ndim))
            del proto_axis_list[self._channel_axis]
            axis = tuple(proto_axis_list)
        reduced = fns.mean(x, axis=axis, keepdims=True)
        self._all_values.append(reduced)
        self._all_shapes.append(tuple(x.shape))

    def _reset(self):
        self._all_values.clear()
        self._all_shapes.clear()

    def _mean_aggregate(self) -> Tensor:
        from nncf.experimental.tensor import functions as fns

        all_values_stack = fns.stack(list(self._all_values))
        return fns.mean(all_values_stack, 0)

    def _shape(self):
        return self._all_shapes[0]

    def _get_statistics(self) -> MeanTensorStatistic:
        return MeanTensorStatistic(self._mean_aggregate(), self._shape())


class RawStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collects tensor samples, where each tensor represented in raw format.
    Each sample stays available for usage in further stages of the algorithm.
    """

    def __init__(self, num_samples: Optional[int] = None) -> None:
        """
        :param num_samples: Optional parameter for statistic collection that regulates
            the number of samples that will be processed.
        """
        super().__init__(num_samples=num_samples)
        self._all_values: List[Tensor] = []

    def _register_input(self, x: Tensor):
        self._all_values.append(x)

    def _reset(self):
        self._all_values.clear()

    def _get_statistics(self) -> RawTensorStatistic:
        return RawTensorStatistic(self._all_values)


class MedianMADStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates median and median absolute deviation (MAD).
    """

    def _prepare_statistics(self) -> Tuple[Tensor, Tensor]:
        first_sample = next(iter(self._samples))
        from nncf.experimental.tensor import functions as fns

        reduction_shape = get_reduction_shape_from_sample_shape(first_sample.shape, self._reduction_axes)
        per_channel_history = get_per_channel_history(self._samples, reduction_shape, discard_zeros=True)
        per_channel_median = [fns.median(channel_hist) for channel_hist in per_channel_history]
        per_channel_mad = []
        for idx, median in enumerate(per_channel_median):
            per_channel_mad.append(fns.median(fns.abs(per_channel_history[idx] - median)))
        median = fns.stack(per_channel_median).reshape(reduction_shape)
        mad = fns.stack(per_channel_mad).reshape(reduction_shape)
        return median, mad

    def _get_statistics(self) -> MedianMADTensorStatistic:
        median, mad = self._prepare_statistics()
        return MedianMADTensorStatistic(median, mad)

    def _register_input(self, x: Tensor):
        self._samples.append(x)


class PercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values of all data history.
    """

    def __init__(
        self,
        percentiles_to_collect: List[float],
        reduction_axes: Optional[ReductionAxes] = None,
        num_samples: int = None,
        window_size: int = None,
    ):
        super().__init__(reduction_axes, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect

    def _prepare_statistics(self) -> Dict[float, Tensor]:
        first_sample = next(iter(self._samples))
        from nncf.experimental.tensor import functions as fns

        reduction_shape = get_reduction_shape_from_sample_shape(first_sample.shape, self._reduction_axes)
        per_channel_history = get_per_channel_history(self._samples, list(reduction_shape))
        percentile_vs_values_dict: Dict[float, Tensor] = {}
        for pc in self._percentiles_to_collect:
            per_channel_percentiles = [fns.quantile(channel_hist, pc / 100) for channel_hist in per_channel_history]
            percentile_vs_values_dict[pc] = fns.stack(per_channel_percentiles).reshape(reduction_shape)
        return percentile_vs_values_dict

    def _register_input(self, x: Tensor):
        self._samples.append(x)

    def _get_statistics(self) -> PercentileTensorStatistic:
        percentile_vs_values_dict = self._prepare_statistics()
        return PercentileTensorStatistic(percentile_vs_values_dict)


class MeanPercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values per step and then averages the results.
    """

    def __init__(
        self,
        percentiles_to_collect: List[float],
        reduction_axes: Optional[ReductionAxes] = None,
        num_samples: int = None,
        window_size: int = None,
    ):
        super().__init__(reduction_axes, num_samples, window_size)
        self._all_pct_values: Dict[float, Deque[Tensor]] = {}
        for pc in percentiles_to_collect:
            self._all_pct_values[pc] = deque(maxlen=window_size)

    def _reset(self):
        for _, val in self._all_pct_values.items():
            val.clear()

    def _register_input(self, x: Tensor):
        for pct, values in self._all_pct_values.items():
            pct_vals = percentile_reduce(x, self._reduction_axes, pct)
            values.append(pct_vals)

    def _get_statistics(self) -> PercentileTensorStatistic:
        mean_percentile_values = {}
        for pct, values in self._all_pct_values.items():
            from nncf.experimental.tensor import functions as fns

            stacked_pct_vals = fns.stack(list(values))
            mean_percentile_values[pct] = fns.mean(stacked_pct_vals, axis=0)
        return PercentileTensorStatistic(mean_percentile_values)
