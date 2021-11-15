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
from typing import Tuple

from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.utils.backend import BackendType

ReductionShape = Tuple[int]


class TensorStatisticCollectorBase(ABC):
    """
    Collector estimate statistics at the quantization point based on the provided reduction shape.
    Statistics can be stored in offline regime and aggregated afterwards or collected on the fly in online regime.
    """

    def __init__(self, reduction_shape: ReductionShape = None, num_samples: int = None):
        self._reduction_shape = reduction_shape
        self._enabled = True
        self._collected_samples = 0
        self._num_samples = num_samples

    def register_input(self, x):
        if not self._enabled or \
                self._num_samples is not None and self._collected_samples >= self._num_samples:
            return x
        if self._reduction_shape is None:
            self._reduction_shape = {tuple(x.shape)}
        self._register_input(x)
        self._collected_samples += 1
        return x

    @abstractmethod
    def _register_input(self, x):
        pass

    def get_statistics(self):
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
        self._collected_samples = 0
        self._reset()

    @abstractmethod
    def _reset(self):
        pass

    def collected_samples(self) -> int:
        return self._collected_samples


class StatisticsNotCollectedError(Exception):
    """Raised when the statistics are not collected but requested"""
    pass


class OnlineTensorStatisticCollector(TensorStatisticCollectorBase, ABC):
    pass


class OfflineTensorStatisticCollector(TensorStatisticCollectorBase, ABC):
    def __init__(self, reduction_shape: ReductionShape = None, num_samples: int = None, window_size: int = None):
        super().__init__(reduction_shape, num_samples)
        self._samples = deque(maxlen=window_size)

    def _reset(self):
        self._samples.clear()


class CollectorParams:
    def __init__(self, is_weights: bool, mode: str, per_channel: bool, init_type: str):
        self._is_weights = is_weights
        self._mode = mode
        self._per_channel = per_channel
        self._init_type = init_type

    @property
    def _use_abs_max(self) -> bool:
        return self._mode == 'symmetric'

    @property
    def _use_means_of_mins(self) -> bool:
        return not self._is_weights and not self._per_channel and self._mode == 'asymmetric'

    @property
    def _use_means_of_maxs(self) -> bool:
        return not self._is_weights and not self._per_channel

    def _get_params_for_min_max(self) -> bool:
        return self._use_abs_max

    def _get_params_for_mixed_min_max(self) -> Tuple[bool, bool, bool]:
        return self._use_abs_max, self._use_means_of_mins, self._use_means_of_maxs

    def _get_params_for_mean_min_max(self) -> bool:
        return self._use_abs_max

    def get_low_level_params_for_collector(self):
        """Generate low-level parameters for collectors"""
        if self._init_type == "min_max":
            return self._get_params_for_min_max()
        if self._init_type == "mixed_min_max":
            return self._get_params_for_mixed_min_max()
        if self._init_type == "mean_min_max":
            return self._get_params_for_mean_min_max()
        raise RuntimeError("Parameters not required or unknown range init type: {}".format(self._init_type))


class MinMaxStatisticCollector(OnlineTensorStatisticCollector, ABC):
    """
    Collector estimates min of minimum values and max of maximum values.
    """

    def __init__(self, use_abs_max: bool, reduction_shape: ReductionShape,
                 num_samples: int = None, nncf_backend = None):
        super().__init__(reduction_shape, num_samples)
        self._use_abs_max = use_abs_max

        self._min_values = None
        self._max_values = None

        if nncf_backend == BackendType.TORCH:
            from nncf.torch.tensor_statistics.aggregation_functions import get_aggregation_function as af
            self._af = af
            self._prefix = 'pt'
        elif nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.quantization.initializers.aggregation_functions import get_aggregation_function as af
            self._af = af
            self._prefix = 'tf'
        else:
            raise RuntimeError('Got an unsupported value of nncf_backend')

    def _register_input_common(self, x):
        min_reduced = self._af(self._prefix + '_reduce_min')(x, self._reduction_shape)
        if self._use_abs_max:
            x = self._af(self._prefix + '_abs')(x)
        max_reduced = self._af(self._prefix + '_reduce_max')(x, self._reduction_shape)

        if self._min_values is None:
            self._min_values = min_reduced
        else:
            self._min_values = self._af(self._prefix + '_min')(min_reduced, self._min_values)

        if self._max_values is None:
            self._max_values = max_reduced
        else:
            self._max_values = self._af(self._prefix + '_max')(max_reduced, self._max_values)

    def _get_statistics(self) -> MinMaxTensorStatistic:
        return MinMaxTensorStatistic(self._min_values, self._max_values)

    def _reset(self):
        self._min_values = None
        self._max_values = None


class MinMaxOfflineStatisticCollectorBase(OfflineTensorStatisticCollector, ABC):
    def __init__(self,
                 use_per_sample_stats: bool,
                 use_abs_max: bool,
                 reduction_shape: ReductionShape,
                 num_samples: int = None,
                 window_size: int = None,
                 nncf_backend: BackendType = None):
        super().__init__(reduction_shape, num_samples)
        self._use_per_sample_stats = use_per_sample_stats
        self._use_abs_max = use_abs_max

        self._all_min_values = deque(maxlen=window_size)
        self._all_max_values = deque(maxlen=window_size)

        if nncf_backend == BackendType.TORCH:
            from nncf.torch.tensor_statistics.aggregation_functions import get_aggregation_function as af
            self._af = af
            self._prefix = 'pt'
        elif nncf_backend == BackendType.TENSORFLOW:
            from nncf.tensorflow.quantization.initializers.aggregation_functions import get_aggregation_function as af
            self._af = af
            self._prefix = 'tf'
        else:
            raise RuntimeError('Got an unsupported value of nncf_backend')

    def _register_input_common(self, x):
        min_reduced = self._af(self._prefix + '_reduce_min')(x, self._reduction_shape)
        if self._use_abs_max:
            x = self._af(self._prefix + '_abs')(x)
        max_reduced = self._af(self._prefix + '_reduce_max')(x, self._reduction_shape)

        if self._use_per_sample_stats:
            self._all_min_values.extend(self._af(self._prefix + '_list_to_extend_stat_history')(min_reduced))
            self._all_max_values.extend(self._af(self._prefix + '_list_to_extend_stat_history')(max_reduced))
        else:
            self._all_min_values.append(min_reduced)
            self._all_max_values.append(max_reduced)

    @abstractmethod
    def _min_aggregate(self):
        pass

    @abstractmethod
    def _max_aggregate(self):
        pass

    def _get_statistics(self) -> MinMaxTensorStatistic:
        return MinMaxTensorStatistic(self._min_aggregate(), self._max_aggregate())

    def _reset(self):
        self._all_min_values.clear()
        self._all_max_values.clear()


class MixedMinMaxStatisticCollector(MinMaxOfflineStatisticCollectorBase, ABC):
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
                 window_size: int = None,
                 nncf_backend: BackendType = None):
        super().__init__(use_per_sample_stats, use_abs_max, reduction_shape, num_samples, window_size, nncf_backend)
        self._use_means_of_mins = use_means_of_mins
        self._use_means_of_maxs = use_means_of_maxs

    def _min_aggregate(self):
        stacked_min = self._af(self._prefix + '_stack')(self._all_min_values)
        if self._use_means_of_mins:
            return self._af(self._prefix + '_mean')(stacked_min, axis=0)
        return self._af(self._prefix + '_tensor_min')(stacked_min, axis=0)

    def _max_aggregate(self):
        stacked_max = self._af(self._prefix + '_stack')(self._all_max_values)
        if self._use_means_of_maxs:
            return self._af(self._prefix + '_mean')(stacked_max, axis=0)
        return self._af(self._prefix + '_tensor_max')(stacked_max, axis=0)


class MeanMinMaxStatisticCollector(MinMaxOfflineStatisticCollectorBase, ABC):
    """
    Collector estimates mean of minimum values and mean of maximum values.
    """

    def _min_aggregate(self):
        stacked_min = self._af(self._prefix + '_stack')(self._all_min_values)
        return self._af(self._prefix + '_mean')(stacked_min, axis=0)

    def _max_aggregate(self):
        stacked_max = self._af(self._prefix + '_stack')(self._all_max_values)
        return self._af(self._prefix + '_mean')(stacked_max, axis=0)
