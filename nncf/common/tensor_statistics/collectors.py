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

from abc import ABC
from abc import abstractmethod
from collections import deque
from typing import Tuple, Optional, List, Union

import numpy as np
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorType
from nncf.common.tensor import TensorElementsType

from nncf.common.tensor_statistics.reduction import get_per_channel_history

ReductionShape = Tuple[int]


class TensorStatisticCollectorBase(ABC):
    """Collector estimate statistics at the quantization point based on the provided reduction shape."""

    def __init__(self, reduction_shape: Optional[ReductionShape] = None, num_samples: Optional[int] = None):
        """
        Initializes Tensor Statistic Collector

        :param reduction_shape: Shape that defines tensor dimensions to reduce.
        :param num_samples: Maximum number of samples to collect.
        """
        self._reduction_shape = reduction_shape
        self._enabled = True
        self._collected_samples = 0
        self._num_samples = num_samples

    def register_input(self, x: TensorType) -> TensorType:
        """Registers input tensor"""
        if not self._enabled or \
                self._num_samples is not None and self._collected_samples >= self._num_samples:
            return x
        if self._reduction_shape is None:
            self._reduction_shape = tuple(range(len(x.shape)))
        self._register_input(x)
        self._collected_samples += 1
        return x

    @abstractmethod
    def _register_input(self, x: TensorType):
        pass

    def get_statistics(self):
        """Returns collected statistics, if present."""
        if self._collected_samples == 0:
            raise StatisticsNotCollectedError()
        return self._get_statistics()

    @abstractmethod
    def _get_statistics(self):
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


class StatisticsNotCollectedError(Exception):
    """Raised when the statistics are not collected but requested."""


class OnlineTensorStatisticCollector(TensorStatisticCollectorBase):
    """Base class for collectors that collects statistics in online regime, without storing the data."""


class OfflineTensorStatisticCollector(TensorStatisticCollectorBase):
    """Collects statistics in offline regime by storing the data and aggregating it afterwards."""

    def __init__(self, reduction_shape: Optional[ReductionShape] = None,
                 num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples)
        self._samples = deque(maxlen=window_size)

    def _reset(self):
        self._samples.clear()


class NNCFCollectorTensorProcessor(ABC):
    """
    An interface of the processing methods for NNCFTensors.
    """

    @staticmethod
    @abstractmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        """
         Computes minimum of elements across dimensions of NNCFTensor.

         :param x: NNCFTensor to reduce
         :param axis: The dimensions to reduce.
         :return: Reduced NNCFTensor.
         """

    @staticmethod
    @abstractmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        """
        Computes maximum of elements across dimensions of NNCFTensor.

        :param x: NNCFTensor to reduce
        :param axis: The dimensions to reduce.
        :return: Reduced NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        """
        Computes the absolute value of a NNCFTensor.

        :param x: NNCFTensor
        :return: Absolute value of a NNCFTensor
        """

    @staticmethod
    @abstractmethod
    def min(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        """
        Returns the min of x1 and x2.

        :param x1: NNCFTensor to compare.
        :param x2: NNCFTensor to compare.
        :return: Compared Tensor.
        """

    @staticmethod
    @abstractmethod
    def max(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        """
        Returns the max of x1 and x2.

        :param x1: NNCFTensor to compare.
        :param x2: NNCFTensor to compare.
        :return: Compared NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def mean(x: NNCFTensor, axis: Union[int, tuple, list]) -> NNCFTensor:
        """
        Computes the mean of elements across given dimensions of NNCFTensor.

        :param x: NNCFTensor to reduce.
        :param axis: The dimensions to reduce.
        :return: Reduced NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def stack(x: NNCFTensor) -> NNCFTensor:
        """
        Stacks a list or deque of NNCFTensors rank-R tensors into one NNCFTensor rank-(R+1) tensor.

        :param x: List or deque of NNCFTensors.
        :param axis: The axis to stack along.
        :return: Stacked NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def unstack(x: NNCFTensor) -> List[NNCFTensor]:
        """
        Unstack a NNCFTensor into list.

        :param x: NNCFTensor to unstack.
        :param axis: The axis to unstack along.
        :return: List of NNCFTensor.
        """

    @staticmethod
    @abstractmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        """
        Returns a sum of each elements in a given NNCFTensor.

        :param tensor: Given NNCFTensor.
        :returns: Sum of each elements of the given NNCFTensor.
        """


class MinMaxStatisticCollector(OnlineTensorStatisticCollector):
    """Collector estimates min of minimum values and max of maximum values."""

    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape, num_samples: int = None):
        super().__init__(reduction_shape, num_samples)
        self._use_abs_max = use_abs_max
        self._tensor_processor = self._get_processor()

        self._min_values = None
        self._max_values = None

    @staticmethod
    @abstractmethod
    def _get_processor():
        pass

    def _register_input_common(self, x: NNCFTensor):
        min_reduced = self._tensor_processor.reduce_min(x, self._reduction_shape)
        if self._use_abs_max:
            x = self._tensor_processor.abs(x)
        max_reduced = self._tensor_processor.reduce_max(x, self._reduction_shape)

        if self._min_values is None:
            self._min_values = min_reduced
        else:
            self._min_values = self._tensor_processor.min(min_reduced, self._min_values)

        if self._max_values is None:
            self._max_values = max_reduced
        else:
            self._max_values = self._tensor_processor.max(max_reduced, self._max_values)

    def _reset(self):
        self._min_values = None
        self._max_values = None


class MinMaxOfflineStatisticCollectorBase(OfflineTensorStatisticCollector):
    """
    Base class for collectors that aggregate statistics
    from minimum and maximum values of tensors.
    """

    def __init__(self,
                 use_per_sample_stats: bool,
                 use_abs_max: bool,
                 reduction_shape: ReductionShape,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shape, num_samples)
        self._use_per_sample_stats = use_per_sample_stats
        self._use_abs_max = use_abs_max
        self._tensor_processor = self._get_processor()

        self._all_min_values = deque(maxlen=window_size)
        self._all_max_values = deque(maxlen=window_size)

    @staticmethod
    @abstractmethod
    def _get_processor():
        pass

    def _register_input_common(self, x: NNCFTensor):
        min_reduced = self._tensor_processor.reduce_min(x, self._reduction_shape)
        if self._use_abs_max:
            x = self._tensor_processor.abs(x)
        max_reduced = self._tensor_processor.reduce_max(x, self._reduction_shape)

        if self._use_per_sample_stats:
            self._all_min_values.extend(self._tensor_processor.unstack(min_reduced))
            self._all_max_values.extend(self._tensor_processor.unstack(max_reduced))
        else:
            self._all_min_values.append(min_reduced)
            self._all_max_values.append(max_reduced)

    @abstractmethod
    def _min_aggregate(self):
        pass

    @abstractmethod
    def _max_aggregate(self):
        pass

    def _reset(self):
        self._all_min_values.clear()
        self._all_max_values.clear()


class MixedMinMaxStatisticCollector(MinMaxOfflineStatisticCollectorBase):
    """
    Collector aggregates (min or mean) of minimum values and (max or mean) of maximum values.
    """

    def __init__(self,
                 use_per_sample_stats: bool,
                 use_abs_max: bool,
                 use_means_of_mins: bool,
                 use_means_of_maxs: bool,
                 reduction_shape: ReductionShape,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(use_per_sample_stats, use_abs_max, reduction_shape, num_samples, window_size)
        self._use_means_of_mins = use_means_of_mins
        self._use_means_of_maxs = use_means_of_maxs

    def _min_aggregate(self):
        stacked_min = self._tensor_processor.stack(self._all_min_values)
        if self._use_means_of_mins:
            return self._tensor_processor.mean(stacked_min, axis=0)
        return self._tensor_processor.reduce_min(stacked_min, axis=0)

    def _max_aggregate(self):
        stacked_max = self._tensor_processor.stack(self._all_max_values)
        if self._use_means_of_maxs:
            return self._tensor_processor.mean(stacked_max, axis=0)
        return self._tensor_processor.reduce_max(stacked_max, axis=0)


class MeanMinMaxStatisticCollector(MinMaxOfflineStatisticCollectorBase):
    """
    Collector aggregates mean of minimum values and mean of maximum values.
    """

    def _min_aggregate(self):
        stacked_min = self._tensor_processor.stack(self._all_min_values)
        return self._tensor_processor.mean(stacked_min, axis=0)

    def _max_aggregate(self):
        stacked_max = self._tensor_processor.stack(self._all_max_values)
        return self._tensor_processor.mean(stacked_max, axis=0)


class MedianMADStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates median and median absolute deviation (MAD).
    """

    def _prepare_statistics(self):
        per_channel_history = get_per_channel_history(self._samples, list(self._reduction_shape),
                                                      discard_zeros=True)
        per_channel_median = [np.median(channel_hist) for channel_hist in per_channel_history]
        per_channel_mad = []
        for idx, median in enumerate(per_channel_median):
            per_channel_mad.append(np.median(abs(per_channel_history[idx] - median)))
        numpy_median = np.asarray(per_channel_median)
        numpy_mad = np.asarray(per_channel_mad)
        return numpy_median, numpy_mad


class PercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values of all data history.
    """

    def __init__(self,
                 percentiles_to_collect: List[float],
                 reduction_shape: Optional[ReductionShape] = None,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._percentiles_to_collect = percentiles_to_collect

    def _prepare_statistics(self):
        per_channel_history = get_per_channel_history(self._samples, list(self._reduction_shape))
        percentile_vs_values_dict = {}
        for pc in self._percentiles_to_collect:
            per_channel_percentiles = [np.percentile(channel_hist, pc) for channel_hist in per_channel_history]
            numpy_percentiles = np.asarray(per_channel_percentiles)
            percentile_vs_values_dict[pc] = numpy_percentiles
        return percentile_vs_values_dict


class MeanPercentileStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector estimates percentile values per step and then averages the results.
    """

    def __init__(self,
                 percentiles_to_collect: List[float],
                 reduction_shape: Optional[ReductionShape] = None,
                 num_samples: int = None,
                 window_size: int = None):
        super().__init__(reduction_shape, num_samples, window_size)
        self._all_pct_values = {}
        for pc in percentiles_to_collect:
            self._all_pct_values[pc] = deque(maxlen=window_size)

    def _reset(self):
        for _, val in self._all_pct_values.items():
            val.clear()
