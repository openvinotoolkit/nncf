"""
 Copyright (c) 2023 Intel Corporation
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, Optional, List, Set, Union, Callable, Dict, Any

import numpy as np
from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorType
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.reduction import get_per_channel_history

ReductionShape = Tuple[int]


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


class TensorReducerBase(ABC):

    def __init__(self,
                 reduction_shape: Optional[ReductionShape] = None,
                 inplace: bool = False):
        self._reduction_shape = reduction_shape
        self._tensor_processor = self._get_processor()
        self._inplace = inplace

    @property
    def inplace(self):
        return self._inplace

    @classmethod
    def name(cls):
        return cls.__name__

    @staticmethod
    @abstractmethod
    def _get_processor():
        pass

    def reduce_input(self, x: TensorType):
        if self.inplace:
            return x

        if self._reduction_shape is None:
            self._reduction_shape = tuple(range(len(x.shape)))
        return self._reduce_out_of_place(x)

    @abstractmethod
    def _reduce_out_of_place(self, x: TensorType):
        pass

    @abstractmethod
    def get_output_name(self, target_node_name: str,
                        port_id: int) -> str:
        pass

    @abstractmethod
    def get_inplace_fn(self):
        pass

    def __hash__(self) -> int:
        return hash(self.name())


class TensorAggregatorBase:
    def __init__(self, reduction_shape, tensor_processor,
                 num_samples: Optional[int]):

        self._reduction_shape = reduction_shape
        self._tensor_processor = tensor_processor
        self._num_samples = num_samples
        self._collected_samples = 0
        self._container = []

    @property
    def num_samples(self) -> int:
        return self._num_samples

    @classmethod
    def name(cls):
        return cls.__name__

    def register_reduced_input(self, x: TensorType):
        if self._num_samples is not None and\
            self._collected_samples >= self._num_samples:
            return
        self._register_reduced_input_impl(x)
        self._collected_samples += 1

    @abstractmethod
    def _register_reduced_input_impl(self, x: TensorType):
        pass

    @abstractmethod
    def aggregate(self):
        pass

    def reset(self):
        self._container = []

    def __hash__(self) -> int:
        return hash((self.name(), self._num_samples))


class TensorCollector:
    def __init__(self,
                 statistic_container: Optional['StatisticContainer'] = None
                 ) -> None:
        self._reducers: Set[TensorReducerBase] = set()
        self._aggregators: Dict[Tuple[int, int], TensorAggregatorBase] = dict()
        self._aggregated_values = None
        self._stat_container_kwargs_map: Dict[str, Tuple[int, int]] = dict()
        self._stat_container = statistic_container
        self._enabled = True

    @property
    def num_samples(self) -> int:
        return max(aggregator.num_samples for aggregator in self._aggregators.values())

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def reducers(self):
        return self._reducers.copy()

    @property
    def aggregators(self):
        return self._aggregators.copy()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def add_branch(self, container_key: str,
                   reducer: TensorReducerBase, aggregator: TensorAggregatorBase) -> None:
        self._reducers.add(reducer)
        key = (hash(reducer), hash(aggregator))
        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        if container_key in self._stat_container_kwargs_map:
            raise RuntimeError(f'Two differend statistics for one'
                               f' container key {container_key} are encountered')
        self._stat_container_kwargs_map[container_key] = key

    def register_input(self, target_node_name: str, port_id,
                        inputs: Dict[str, TensorType]):
        if not self._enabled:
            return

        reduced_inputs = {}
        for reducer in self._reducers:
            input_name = reducer.get_output_name(target_node_name, port_id)
            reduced_input = reducer.reduce_input(inputs[input_name])
            reduced_inputs[hash(reducer)] = reduced_input

        for (reducer_hash, _), aggregator, in self._aggregators.items():
            aggregator.register_reduced_input(reduced_inputs[reducer_hash])

    def _aggregate(self) -> None:
        result = {}
        for key, aggregator, in self._aggregators.items():
            val = aggregator.aggregate()
            result[key] = val
        return result

    def get_statistics(self):
        aggregated_values = self._aggregate()
        kwargs = {}
        for container_key, branch_key in self._stat_container_kwargs_map.items():
            kwargs[container_key] = aggregated_values[branch_key]

        if not self._stat_container:
            return kwargs
        return self._stat_container(**kwargs)

    def get_inplace_fn(self):
        retval = []
        for reducer in self._reducers:
            if reducer.inplace:
                retval.append(reducer.get_inplace_fn())
        return retval

    def any_stat_out_of_place(self) -> bool:
        return any(not reducer.inplace for reducer in self._reducers)

    def replace_aggregator(self, key, aggregator):
        assert key in self._aggregators
        assert key[1] == hash(aggregator)
        self._aggregators[key] = aggregator

    def reset(self):
        for aggregator in self._aggregators.values:
            aggregator.reset()


class MergedTensorCollector(TensorCollector):
    def __init__(self, tensor_collectors: List[TensorCollector]) -> None:
        super().__init__()
        aggregators: Dict[Tuple[int, int], List[Tuple[TensorCollector, TensorAggregatorBase]]] =\
            defaultdict(list)
        for tensor_collector in tensor_collectors:
            if not tensor_collector.enabled:
                continue
            self._reducers.update(tensor_collector.reducers)
            for key, aggregator in tensor_collector.aggregators.items():
                aggregators[key].append((tensor_collector, aggregator))

        for key, aggregators_to_merge in aggregators.items():
            _, unique_aggregator = aggregators_to_merge[0]
            for tensor_collector, _ in aggregators_to_merge[1:]:
                tensor_collector.replace_aggregator(key, unique_aggregator)
            self._aggregators[key] = unique_aggregator


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType):
        return self._tensor_processor.reduce_min(x, self._reduction_shape)


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType):
        return self._tensor_processor.reduce_max(x, self._reduction_shape)


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType):
        x = self._tensor_processor.abs(x)
        return self._tensor_processor.reduce_max(x, self._reduction_shape)


class OnlineMinAggregator(TensorAggregatorBase):
    def _register_reduced_input_impl(self, x: TensorType):
        if not self._container:
            self._container = x
        else:
            self._container = self._tensor_processor.min(x, self._container)

    def aggregate(self):
        return self._container.tensor


class OnlineMaxAggregator(TensorAggregatorBase):
    def _register_reduced_input_impl(self, x: TensorType):
        if not self._container:
            self._container = x
        else:
            self._container = self._tensor_processor.max(x, self._container)

    def aggregate(self):
        return self._container.tensor


class OfflineMinMaxAggregatorBase(TensorAggregatorBase):
    def __init__(self, reduction_shape, tensor_processor, use_per_sample_stats: bool,
                 num_samples: Optional[int], window_size=None):
        super().__init__(reduction_shape, tensor_processor, num_samples)
        self._window_size = window_size
        self._container = deque(maxlen=window_size)
        self._use_per_sample_stats = use_per_sample_stats

    def _register_reduced_input_impl(self, x: TensorType):
        if self._use_per_sample_stats:
            self._container.extend(self._tensor_processor.unstack(x))
        else:
            self._container.append(x)


class OfflineMinAggregator(OfflineMinMaxAggregatorBase):
    def aggregate(self):
        stacked_val = self._tensor_processor.stack(self._container)
        return self._tensor_processor.reduce_min(stacked_val, axis=0).tensor


class OfflineMaxAggregator(OfflineMinMaxAggregatorBase):
    def aggregate(self):
        stacked_val = self._tensor_processor.stack(self._container)
        return self._tensor_processor.reduce_max(stacked_val, axis=0).tensor


class OfflineMeanAggregator(OfflineMinMaxAggregatorBase):
    def aggregate(self):
        stacked_val = self._tensor_processor.stack(self._container)
        return self._tensor_processor.mean(stacked_val, axis=0).tensor


class StatisticsNotCollectedError(Exception):
    """Raised when the statistics are not collected but requested."""


class OnlineTensorStatisticCollector(TensorReducerBase):
    """Base class for collectors that collects statistics in online regime, without storing the data."""


class OfflineTensorStatisticCollector(TensorReducerBase):
    """Collects statistics in offline regime by storing the data and aggregating it afterwards."""

    def __init__(self, reduction_shape: Optional[ReductionShape] = None, window_size: Optional[int] = None):
        super().__init__(reduction_shape)
        self._samples = deque(maxlen=window_size)

    def reset(self):
        self._samples.clear()


class MinMaxStatisticCollector(OnlineTensorStatisticCollector):
    """Collector estimates min of minimum values and max of maximum values."""

    def __init__(self, reduction_shape: ReductionShape, use_abs_max: bool):
        super().__init__(reduction_shape)
        self._use_abs_max = use_abs_max

        self._min_values = None
        self._max_values = None

    def reduce_input(self, x: NNCFTensor):
        min_reduced = self._tensor_processor.reduce_min(x, self._reduction_shape)
        if self._use_abs_max:
            x = self._tensor_processor.abs(x)
        max_reduced = self._tensor_processor.reduce_max(x, self._reduction_shape)
        return min_reduced, max_reduced

    def register_reduced_input(self, min_reduced: NNCFTensor, max_reduced: NNCFTensor):
        if self._min_values is None:
            self._min_values = min_reduced
        else:
            self._min_values = self._tensor_processor.min(min_reduced, self._min_values)

        if self._max_values is None:
            self._max_values = max_reduced
        else:
            self._max_values = self._tensor_processor.max(max_reduced, self._max_values)

    def reset(self):
        self._min_values = None
        self._max_values = None


@dataclass
class MinMaxOfflineStatisticCollectorSpec:
    use_per_sample_stats: bool
    use_abs_max: bool
    reduction_shape: ReductionShape
    num_samples: int = None
    window_size: int = None
    inplace: bool = False

class MinMaxOfflineStatisticCollectorBase(OfflineTensorStatisticCollector):
    """
    Base class for collectors that aggregate statistics
    from minimum and maximum values of tensors.
    """

    def __init__(self,
                 use_per_sample_stats: bool, use_abs_max: bool,
                 reduction_shape: Optional[ReductionShape] = None,
                 window_size: Optional[int] = None):
        super().__init__(reduction_shape, window_size)
        self._use_per_sample_stats = use_per_sample_stats
        self._use_abs_max = use_abs_max

        self._all_min_values = deque(maxlen=window_size)
        self._all_max_values = deque(maxlen=window_size)

    def reduce_input(self, x: NNCFTensor):
        min_reduced = self._tensor_processor.reduce_min(x, self._reduction_shape)
        if self._use_abs_max:
            x = self._tensor_processor.abs(x)
        max_reduced = self._tensor_processor.reduce_max(x, self._reduction_shape)
        return min_reduced, max_reduced

    def register_reduced_input(self, min_reduced: NNCFTensor, max_reduced: NNCFTensor):
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

    def reset(self):
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
        super().__init__(use_per_sample_stats, use_abs_max,
                         reduction_shape, num_samples, window_size)
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


class MeanStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collector that aggregates statistics as mean along a pre-assigned axis.
    """

    def __init__(self,
                 tensor_processor: NNCFCollectorTensorProcessor,
                 reduction_shape: ReductionShape,
                 num_samples: Optional[int] = None,
                 window_size: Optional[int] = None) -> None:
        """
        :param reduction_shape: The shape for the reduction while statistics collection.
            For the MeanStatisticCollector this parameter contains the main axis.
        :param num_samples: Optional parameter for statistic collection that regulates
            the number of samples that will be processed.
        :param window_size: Optional maximum length for the statistic collection
        """
        super().__init__(tensor_processor, reduction_shape, num_samples)
        self._all_values = deque(maxlen=window_size)
        self._all_shapes = deque(maxlen=window_size)

    def reduce_input(self, x: NNCFTensor):
        if self._reduction_shape == 0:
            return (self._tensor_processor.batch_mean(x),)
        else:
            return (self._tensor_processor.mean_per_channel(x, self._reduction_shape),)

    def register_reduced_input(self, reduced_input: NNCFTensor):
        self._all_values.append(reduced_input)
        self._all_shapes.append(reduced_input.shape)

    def reset(self):
        self._all_values.clear()
        self._all_shapes.clear()

    def _mean_aggregate(self):
        all_values_stack = self._tensor_processor.stack(self._all_values)
        return self._tensor_processor.mean(all_values_stack, 0)

    def _shape(self):
        return self._all_shapes[0]


class BatchStatisticCollector(OfflineTensorStatisticCollector):
    """
    Collects tensor samples, where each tensor is averaged along the batch axis (and only that axis).
    Each sample stays available for usage in further stages of the algorithm.
    """

    def __init__(self, tensor_processor: NNCFCollectorTensorProcessor) -> None:
        """
        :param num_samples: Optional parameter for statistic collection that regulates
            the number of samples that will be processed.
        """
        super().__init__(tensor_processor)
        self._all_values = []

    def _reduce_input(self, x: NNCFTensor):
        return (self._tensor_processor.batch_mean(x),)

    def _register_reduced_input(self, reduced_input: NNCFTensor):
        self._all_values.append(reduced_input.tensor)

    def _reset(self):
        self._all_values.clear()


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
