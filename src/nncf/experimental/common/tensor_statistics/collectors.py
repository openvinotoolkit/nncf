# Copyright (c) 2025 Intel Corporation
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
from collections import defaultdict
from collections import deque
from copy import deepcopy
from typing import Any, Optional, TypeVar, Union

import nncf
import nncf.tensor.functions as fns
from nncf.common.tensor import TensorType
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.experimental.common.tensor_statistics.statistical_functions import mean_per_channel
from nncf.experimental.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import TensorStatistic
from nncf.quantization.advanced_parameters import AggregatorType
from nncf.quantization.range_estimator import StatisticsType
from nncf.tensor import Tensor

InplaceInsertionFNType = TypeVar("InplaceInsertionFNType")
AggregationAxes = tuple[int, ...]


class TensorReducerBase(ABC):
    """
    Tensor reducer is a callable object that reduces tensors according to
    the specified rule. Could handle tensors inplace or out of place.
    """

    def __init__(self, reduction_axes: Optional[ReductionAxes] = None, inplace: bool = False):
        """
        :param reduction_axes: Reduction axes for reduction calculation. Equal to list(range(len(input.shape)))
            if empty.
        :param inplace: Whether should be calculated inplace or out of place.
        """
        self._reduction_axes = reduction_axes
        self._inplace = inplace
        self._keepdims = True

    @property
    def inplace(self):
        return self._inplace

    @property
    def output_port_id(self) -> int:
        """
        Port id of the last node of the reducer subgraph if statistic is inplace.
        Port id of the reducer output return node if statistic is not inplace.
        """
        return 0

    @property
    def name(self):
        return self.__class__.__name__ + str(self.__hash__())

    @abstractmethod
    def _reduce_out_of_place(self, x: list[TensorType]) -> list[TensorType]:
        """
        Specifies the reduction rule.

        :param x: Tensor to register.
        """

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        """
        Returns correspondent inplace operation builder if inplace operations are available in backend.

        :return: Inplace operation builder if possible else None.
        """
        return None

    def __call__(self, x: list[Tensor]):
        if any(t.isempty() for t in x):
            return None

        if self.inplace:
            return x

        return self._reduce_out_of_place(x)

    def __eq__(self, __o: object) -> bool:
        return (
            isinstance(__o, self.__class__)
            and self._reduction_axes == __o._reduction_axes
            and self._inplace == __o.inplace
        )

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes))

    def _get_reduction_axes(self, tensor: Tensor) -> ReductionAxes:
        if self._reduction_axes is not None:
            return self._reduction_axes
        return tuple(range(len(tensor.shape)))


class AggregatorBase:
    """
    Aggregator is designed to receive (register) calculated statistics and aggregate them.
    """

    def __init__(
        self,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        """
        :param aggregation_axes: Axes along which to operate.
            Registered statistics are stacked along zero axis,
            axes >=1 correspond to received statistic axes shifted left by 1.
        :param num_samples: Maximum number of samples to collect. Aggregator
            skips tensor registration if tensor registration was called num_samples times before.
            Aggregator never skips registration if num_samples is None.
        :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
        """
        self._aggregation_axes = (0,) if aggregation_axes is None else aggregation_axes
        self._keepdims = True
        self._num_samples = num_samples
        self._collected_samples = 0
        self._window_size = window_size
        self._container = deque(maxlen=window_size)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def register_reduced_input(self, x: TensorType):
        if self._num_samples is not None and self._collected_samples >= self._num_samples:
            return
        self._register_reduced_input_impl(x)
        self._collected_samples += 1

    @abstractmethod
    def _register_reduced_input_impl(self, x: TensorType) -> None:
        """
        Registers incoming tensor in tensor aggregator.

        :param x: Tensor to register.
        """

    def aggregate(self) -> Any:
        """
        Aggregates collected tensors and returns aggregated result.
        In case no tensors were collected returns None.

        :return: Aggregated result.
        """
        if self._collected_samples:
            return self._aggregate_impl()
        return None

    @abstractmethod
    def _aggregate_impl(self) -> Any:
        """
        Aggregates collected tensors and returns aggregated result.

        :return: Aggregated result.
        """

    def reset(self):
        self._collected_samples = 0
        self._container = []

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and self._num_samples == __o.num_samples

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


class TensorCollector:
    """
    Calculates statistics at given tensors according to registered statistic branches.
    Statistic branch consists of one reducer and one aggregator instance. TensorCollector
    applies a reducer on a correspondent inputs and then passes the one of the reduced tensors
    chosen by output port id to a correspondent aggregator for each registered statistic branch.
    Receives tensors by `register_input` method. Aggregated values as a TensorStatistic instance or
    a dict could be collected by `get_statistics` call.
    """

    def __init__(self, statistic_container: Optional[type[TensorStatistic]] = None) -> None:
        self._reducers: set[TensorReducerBase] = set()
        self._aggregators: dict[tuple[int, int, int], AggregatorBase] = {}
        self._stat_container_kwargs_map: dict[str, tuple[int, int, int]] = {}
        self._stat_container = statistic_container
        self.enable()
        self.clear_cache()

    @property
    def num_samples(self) -> Optional[int]:
        output = None
        for aggregator in self._aggregators.values():
            if aggregator.num_samples and output:
                output = max(output, aggregator.num_samples)
            else:
                output = aggregator.num_samples
        return output

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

    def register_statistic_branch(
        self,
        container_key: str,
        reducer: TensorReducerBase,
        aggregator: AggregatorBase,
        reducer_output_port_id: int = 0,
    ) -> None:
        """
        Registers statistic collection branch for a container key. Correspondent input will be reduced
        by given reducer and reduced value will be registered and aggregated by given aggregator.
        Passed container key should be unique for the TensorCollector instance.
        Passed aggregator instance should never be used twice for one TensorCollector instance.

        :param container_key: Container key to pass aggregated statistic to.
        :param reducer: TensorReducer instance for the statistic collection branch.
        :param aggregator: TensorAggregator instance for the statistic collection branch.
        :reducer_output_port_id: Reducer target output port id.
        """
        if container_key in self._stat_container_kwargs_map:
            msg = f"Two different statistic branches for one container key {container_key} are encountered"
            raise nncf.InternalError(msg)
        if any(aggr is aggregator for aggr in self._aggregators.values()):
            msg = f"One aggregator instance {aggregator} for different branches is encountered"
            raise nncf.InternalError(msg)

        self._reducers.add(reducer)
        key = (hash(reducer), reducer_output_port_id, hash(aggregator))

        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        self._stat_container_kwargs_map[container_key] = key

    def register_inputs(self, inputs: dict[int, list[Tensor]]) -> None:
        """
        Registers given input in TensorCollector.

        :param inputs: Tensor inputs in format of dict where keys
            are reducer names and values are correspondent input tensors
        """
        if not self.enabled:
            return
        reduced_inputs = {}
        for reducer in self._reducers:
            reducer_hash = hash(reducer)
            input_ = inputs[reducer_hash]
            reduced_input = reducer(input_)
            if reduced_input is not None:
                reduced_inputs[reducer_hash] = reduced_input

        for (
            (reducer_hash, reducer_port_id, _),
            aggregator,
        ) in self._aggregators.items():
            if reducer_hash in reduced_inputs:
                aggregator.register_reduced_input(reduced_inputs[reducer_hash][reducer_port_id])

    def register_input_for_all_reducers(self, input_: Tensor) -> None:
        """
        Registers given input_ in each available statistic collection branch.

        :param input_: Tensor input to register.
        """
        self.register_inputs({hash(reducer): [input_] for reducer in self._reducers})

    def _aggregate(self) -> None:
        result = {}
        for (
            key,
            aggregator,
        ) in self._aggregators.items():
            val = aggregator.aggregate()
            result[key] = val
        return result

    def set_cache(self, statistics: TensorStatistic) -> None:
        """
        Sets cached statistics from given config and disable TensorCollector.
        :param statistics: TensorStatistic.
        """
        self._cached_statistics = statistics
        self.reset()
        self.disable()

    def create_statistics_container(self, config: dict[str, Any]) -> TensorStatistic:
        """
        Returns a TensorStatistic instance with aggregated values.

        :param config: Aggregated values.
        :return: TensorStatistic instance.
        """
        if not self._stat_container:  # TODO(kshpv): need to remove an ability to return a Dict.
            return config
        return self._stat_container.from_config(config)

    def clear_cache(self) -> None:
        """
        Clears the cached statistics and enables TensorCollector.
        """
        self._cached_statistics = None

    def get_statistics(self) -> TensorStatistic:
        """
        Returns aggregated values in format of a TensorStatistic instance or
        a dict.

        :return: Aggregated values.
        """
        if self._cached_statistics is not None:
            return deepcopy(self._cached_statistics)

        aggregated_values = self._aggregate()
        statistics_config = {}
        for container_key, branch_key in self._stat_container_kwargs_map.items():
            statistics_config[container_key] = aggregated_values[branch_key]
        return self.create_statistics_container(statistics_config)

    def replace_aggregator(self, key: tuple[int, int, int], aggregator: AggregatorBase) -> None:
        """
        Friend method that replaces aggregator instance on equivalent one.
        Key should be valid for for given aggregator and a statistic branch
        with key should be present in TensorCollector.

        :param key: Statistic branch key.
        :param aggregator: Aggregator instance to replace existing instance by given key.
        """
        assert key in self._aggregators
        assert key[2] == hash(aggregator)
        self._aggregators[key] = aggregator

    def reset(self):
        for aggregator in self._aggregators.values():
            aggregator.reset()

    @staticmethod
    def get_tensor_collector_inputs(
        outputs: dict[str, Tensor], output_info: list[tuple[int, list[str]]]
    ) -> dict[int, list[Tensor]]:
        """
        Static method that converts all model outputs and collected output_info
        to a layout required for `register_inputs` method. This method is not a part of
        `register_inputs` to avoid all inputs passing to `TensorCollector.register_inputs` method.

        :param outputs: Target model outputs.
        :param output_info: Output info collected by a `TensorCollector.get_output_info` method.
        :returns: Model outputs in a format required by `TensorCollector.register_inputs` method.
        """
        target_inputs = {}
        for reducer, names in output_info:
            target_inputs[reducer] = [outputs[name] for name in names]
        return target_inputs


class MergedTensorCollector(TensorCollector):
    """
    Tensor collector that merge several tensor collectors in one.
    Statistics collected by a merged tensor collector automatically available
    in all tensor collectors that were merged by the merged tensor collector.
    This works because merged tensor collectors share tensor aggregators instances with
    the merged tensor collector.
    """

    def __init__(self, tensor_collectors: list[TensorCollector]) -> None:
        """
        :param tensor_collectors: Tensor collectors to merge.
        """
        super().__init__()
        aggregators: dict[tuple[int, int, int], list[tuple[TensorCollector, AggregatorBase]]] = defaultdict(list)
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


##################################################
# Reducers
##################################################


class RawReducer(TensorReducerBase):
    def __init__(self):
        super().__init__(inplace=False)

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        return x


class ShapeReducer(TensorReducerBase):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace=inplace)

    def _reduce_out_of_place(self, x: list[TensorType]) -> list[tuple[int, ...]]:
        return [x[0].shape]

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [fns.min(x, reduction_axes, keepdims=self._keepdims)]


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [fns.max(x, reduction_axes, keepdims=self._keepdims)]


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = fns.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        return [fns.max(x, reduction_axes, keepdims=self._keepdims)]


class MeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [fns.mean(x, reduction_axes, keepdims=self._keepdims)]


class MeanVarianceReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        variance = fns.var(x, reduction_axes)
        return [fns.mean(variance)]


class MaxVarianceReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        variance = fns.var(x, reduction_axes)
        return [fns.max(variance)]


class MeanAbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = fns.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        abs_max = fns.max(x, reduction_axes, keepdims=self._keepdims)
        return [fns.mean(abs_max)]


class QuantileReducerBase(TensorReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        quantile: Optional[Union[float, tuple[float]]] = None,
        inplace: bool = False,
    ):
        super().__init__(reduction_axes=reduction_axes, inplace=False)
        self._quantile = (0.01, 0.99) if quantile is None else quantile

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, tuple(self._quantile)))


class QuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return fns.quantile(x, self._quantile, reduction_axes, keepdims=self._keepdims)


class AbsQuantileReducer(QuantileReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        quantile: Optional[Union[float, list[float]]] = None,
        inplace: bool = False,
    ):
        quantile = (0.99,) if quantile is None else quantile
        super().__init__(reduction_axes=reduction_axes, quantile=quantile, inplace=False)

    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        x = fns.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        return fns.quantile(x, self._quantile, reduction_axes, keepdims=self._keepdims)


class BatchMeanReducer(TensorReducerBase):
    def __init__(self, inplace: bool = False):
        super().__init__(None, inplace)

    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        return [fns.mean(x[0], axis=0, keepdims=True)]


class MeanPerChReducer(TensorReducerBase):
    def __init__(self, channel_axis: int = 1, inplace: bool = False):
        super().__init__(inplace=inplace)
        self._channel_axis = channel_axis

    def _reduce_out_of_place(self, x: list[Tensor]) -> list[Tensor]:
        return [mean_per_channel(x[0], self._channel_axis)]

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._channel_axis == __o._channel_axis

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, self._channel_axis))


##################################################
# Aggregators
##################################################


class NoopAggregator(AggregatorBase):
    def __init__(self, num_samples: Optional[int]):
        super().__init__(None, num_samples=num_samples)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container.append(x)

    def _aggregate_impl(self):
        return self._container


class ShapeAggregator(AggregatorBase):
    def __init__(self):
        super().__init__(None, num_samples=1)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container = x

    def _aggregate_impl(self):
        return self._container.shape


class OnlineAggregatorBase(AggregatorBase, ABC):
    """
    Base class for aggregators which are using aggregation function fn with following property:
    fn([x1, x2, x3]) == fn([fn([x1, x2]), x3]) where x1, x2, x3 are samples to aggregate.
    Online aggregation fn([fn([x1, x2]), x3]) allows to keep memory stamp low as only
    one sample is stored during statistic collection.
    """

    def _register_reduced_input_impl(self, x: Tensor) -> None:
        online_aggregation_axes = tuple(dim - 1 for dim in self._aggregation_axes if dim != 0)
        if online_aggregation_axes:
            reduced = self._aggregation_fn(x, axis=online_aggregation_axes, keepdims=self._keepdims)
        else:
            reduced = x
        if 0 in self._aggregation_axes:
            stacked_tensors = fns.stack([reduced, *self._container], axis=0)
            aggregated = self._aggregation_fn(stacked_tensors, axis=0, keepdims=self._keepdims)
            aggregated = fns.squeeze(aggregated, 0)
            self._container = [aggregated]
        else:
            self._container.append(reduced)

    def _aggregate_impl(self) -> Tensor:
        if 0 in self._aggregation_axes and self._keepdims:
            return self._container[0]
        return fns.stack(self._container)

    @abstractmethod
    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        pass


class MinAggregator(OnlineAggregatorBase):
    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        return fns.min(stacked_value, axis=axis, keepdims=keepdims)


class MaxAggregator(OnlineAggregatorBase):
    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        return fns.max(stacked_value, axis=axis, keepdims=keepdims)


class OfflineAggregatorBase(AggregatorBase, ABC):
    """
    Base class for aggregators which are using aggregation function fn which
    does not fulfill property fn([x1, x2, x3]) == fn([fn([x1, x2]), x3])
    where x1, x2, x3 are samples to aggregate. Child aggregators collect
    all samples in a container and aggregate them in one step.
    """

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container.append(x)

    def _aggregate_impl(self) -> Tensor:
        # Case when all registered tensors have identical shape
        if all(self._container[0].shape == x.shape for x in self._container):
            stacked_value = fns.stack(self._container)
            aggregated = self._aggregation_fn(stacked_value, axis=self._aggregation_axes, keepdims=self._keepdims)
            return fns.squeeze(aggregated, 0)
        online_axes = tuple(x - 1 for x in self._aggregation_axes if x > 0)

        # Case when some registered tensors have different shapes and
        # 0 is present in the aggregation axes
        if 0 in self._aggregation_axes:
            stacked_value, shape_after_aggregation = _move_axes_flatten_cat(self._container, online_axes)
            aggregated = self._aggregation_fn(stacked_value, axis=0, keepdims=False)
            if self._keepdims:
                aggregated = fns.reshape(aggregated, shape_after_aggregation)
            return aggregated

        # Case when some registered tensors have different shapes and
        # 0 is not present in the aggregation axes
        ret_val = []
        for tensor in self._container:
            ret_val.append(self._aggregation_fn(tensor, axis=online_axes, keepdims=self._keepdims))
        return fns.stack(ret_val, axis=0)

    @abstractmethod
    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        pass


class MeanAggregator(OfflineAggregatorBase):
    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        return fns.mean(stacked_value, axis=axis, keepdims=keepdims)


class MedianAggregator(OfflineAggregatorBase):
    def _aggregation_fn(self, stacked_value: Tensor, axis: AggregationAxes, keepdims: bool) -> Tensor:
        return fns.median(stacked_value, axis=axis, keepdims=keepdims)


class NoOutliersAggregatorBase(OfflineAggregatorBase, ABC):
    def __init__(
        self,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
        quantile: float = 0.01,
    ):
        super().__init__(aggregation_axes=aggregation_axes, num_samples=num_samples)
        self._window_size = window_size
        self._container = deque(maxlen=window_size)
        self._quantile = quantile

    def _aggregation_fn(self, stacked_value: Tensor, axis: int, keepdims: bool) -> Tensor:
        low_values, high_values = fns.quantile(stacked_value, q=(self._quantile, 1 - self._quantile), axis=axis)
        outliers_mask = fns.logical_or(stacked_value < low_values, high_values < stacked_value)
        aggregated = self._masked_aggregation_fn(
            stacked_samples=stacked_value,
            mask=outliers_mask,
            axis=axis,
            keepdims=keepdims,
        )
        return aggregated

    @abstractmethod
    def _masked_aggregation_fn(
        self, stacked_samples: Tensor, mask: Tensor, axis: AggregationAxes, keepdims: bool
    ) -> Tensor:
        pass

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._quantile))


class MeanNoOutliersAggregator(NoOutliersAggregatorBase):
    def _masked_aggregation_fn(
        self, stacked_samples: Tensor, mask: Tensor, axis: AggregationAxes, keepdims: bool
    ) -> Tensor:
        return fns.masked_mean(stacked_samples, axis=axis, mask=mask, keepdims=keepdims)


class MedianNoOutliersAggregator(NoOutliersAggregatorBase):
    def _masked_aggregation_fn(
        self, stacked_samples: Tensor, mask: Tensor, axis: AggregationAxes, keepdims: bool
    ) -> Tensor:
        return fns.masked_median(stacked_samples, axis=axis, mask=mask, keepdims=keepdims)


class MedianAbsoluteDeviationAggregator(AggregatorBase):
    def __init__(
        self,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
    ):
        super().__init__(
            aggregation_axes=aggregation_axes,
            num_samples=num_samples,
            window_size=window_size,
        )
        if 0 not in self._aggregation_axes:
            msg = "Aggregation without 0 dim is not supported yet for MedianAbsoluteDeviationAggregator"
            raise NotImplementedError(msg)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        return self._container.append(x)

    def _aggregate_impl(self) -> dict[str, Tensor]:
        stacked_val, shape_after_aggregation = _move_axes_flatten_cat(
            self._container, [x - 1 for x in self._aggregation_axes if x > 0]
        )

        mask = fns.abs(stacked_val) < fns.finfo(stacked_val).eps
        median_per_ch = fns.masked_median(stacked_val, mask=mask, axis=0, keepdims=True)

        mad_values = fns.median(
            fns.abs(stacked_val - median_per_ch),
            axis=0,
            keepdims=False,
        )
        if self._keepdims:
            median_per_ch = fns.reshape(median_per_ch, shape_after_aggregation)
            mad_values = fns.reshape(mad_values, shape_after_aggregation)
        else:
            median_per_ch = fns.squeeze(median_per_ch, 0)
        return {
            MedianMADTensorStatistic.MEDIAN_VALUES_STAT: median_per_ch,
            MedianMADTensorStatistic.MAD_VALUES_STAT: mad_values,
        }


class PercentileAggregator(AggregatorBase):
    def __init__(
        self,
        percentiles_to_collect: list[float],
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
    ):
        super().__init__(aggregation_axes=aggregation_axes, num_samples=num_samples)
        if 0 not in self._aggregation_axes:
            msg = "Aggregation without 0 dim is not supported yet for PercentileAggregator"
            raise NotImplementedError(msg)
        self._percentiles_to_collect = percentiles_to_collect
        self._window_size = window_size
        self._container = deque(maxlen=window_size)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        return self._container.append(x)

    def _aggregate_impl(self) -> dict[float, Tensor]:
        stacked_val, shape_after_aggregation = _move_axes_flatten_cat(
            self._container, [x - 1 for x in self._aggregation_axes if x > 0]
        )

        percentiles = fns.percentile(stacked_val, self._percentiles_to_collect, axis=0, keepdims=False)
        retval = {}
        for idx, percentile in enumerate(self._percentiles_to_collect):
            value = percentiles[idx]
            if self._keepdims:
                value = fns.reshape(value, shape_after_aggregation)
            retval[percentile] = value
        return retval


class HAWQAggregator(AggregatorBase):
    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples=num_samples)
        self._container = Tensor(0.0)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        trace = fns.sum(fns.multiply(x, x))
        # NOTE: average trace?? divide by number of diagonal elements
        # TODO: revise this formula as possibly it is with an error; adopted from previous HAWQ implementation
        self._container = (self._container + trace) / x.size

    def _aggregate_impl(self) -> Tensor:
        return self._container * 2 / self._collected_samples


def _move_axes_flatten_cat(
    tensor_list: list[Tensor], aggregation_axes: tuple[int, ...]
) -> tuple[Tensor, tuple[int, ...]]:
    """
    Moves aggregation axes to the beginning of the tensor shape for each tensor from the list, flattens
    and concatenates them in 0 dimension. Computes target shape for the processed tensor
    after an aggregation function is applied to it. Target shape preserves original order
    of dimensions and replaces aggregated dimensions by 1.

    :param tensor_list: Tensor list to process.
    :param aggregation_axes: Aggregation axes to move, flatten and concatenate.
    :return: Tuple of the processed tensor and
        target shape for the processed tensor after an aggregation function is applied to it.
    """
    tensor_shape = list(tensor_list[0].shape)

    # Transpose dims to move aggregation axes forward
    transpose_dims = list(range(len(tensor_shape)))
    for idx, axis in enumerate(aggregation_axes):
        transpose_dims[axis], transpose_dims[idx] = transpose_dims[idx], transpose_dims[axis]

    # Shape to flatten aggregation axes
    reshape_shape = [-1] + [tensor_shape[dim] for dim in transpose_dims][len(aggregation_axes) :]

    reshaped_tensors = []
    for tensor in tensor_list:
        transposed_t = fns.transpose(tensor, transpose_dims)
        reshaped_tensors.append(fns.reshape(transposed_t, reshape_shape))

    shape_after_aggregation = tuple(1 if idx in aggregation_axes else dim for idx, dim in enumerate(tensor_shape))
    return fns.concatenate(reshaped_tensors, axis=0), shape_after_aggregation


REDUCERS_MAP = {
    StatisticsType.MIN: MinReducer,
    StatisticsType.MAX: MaxReducer,
    StatisticsType.ABS_MAX: AbsMaxReducer,
    StatisticsType.MEAN: MeanReducer,
    StatisticsType.QUANTILE: QuantileReducer,
    StatisticsType.ABS_QUANTILE: AbsQuantileReducer,
}

AGGREGATORS_MAP = {
    AggregatorType.MIN: MinAggregator,
    AggregatorType.MAX: MaxAggregator,
    AggregatorType.MEAN: MeanAggregator,
    AggregatorType.MEAN_NO_OUTLIERS: MeanNoOutliersAggregator,
    AggregatorType.MEDIAN: MedianAggregator,
    AggregatorType.MEDIAN_NO_OUTLIERS: MedianNoOutliersAggregator,
}


# class HistogramObserver(UniformQuantizationObserverBase):
#    r"""
#    The module records the running histogram of tensor values along with
#    min/max values. ``calculate_qparams`` will calculate scale and zero_point.
#
#    Args:
#        bins: Number of bins to use for the histogram
#        dtype: dtype argument to the `quantize` node needed to implement the
#               reference model spec
#        qscheme: Quantization scheme to be used
#        reduce_range: Reduces the range of the quantized data type by 1 bit
#        eps: Epsilon value for float32, Defaults to `torch.finfo(torch.float32).eps`.
#
#    The scale and zero point are computed as follows:
#
#    1. Create the histogram of the incoming inputs.
#        The histogram is computed continuously, and the ranges per bin change
#        with every new tensor observed.
#    2. Search the distribution in the histogram for optimal min/max values.
#        The search for the min/max values ensures the minimization of the
#        quantization error with respect to the floating point model.
#    3. Compute the scale and zero point the same way as in the
#        :class:`~torchao.quantization.pt2e.MinMaxObserver`
#    """
#
#    histogram: torch.Tensor
#    min_val: torch.Tensor
#    max_val: torch.Tensor
#
#    def __init__(
#        self,
#        bins: int = 2048,
#        dtype: torch.dtype = torch.quint8,
#        qscheme=torch.per_tensor_affine,
#        reduce_range=False,
#        quant_min=None,
#        quant_max=None,
#        factory_kwargs=None,
#        eps=torch.finfo(torch.float32).eps,
#        is_dynamic=False,
#        **kwargs,
#    ) -> None:
#        if not is_per_tensor(qscheme):
#            raise NotImplementedError(
#                "HistogramObserver's qscheme only support torch.per_tensor_symmetric \
#                    and torch.per_tensor_affine."
#            )
#        if is_dynamic:
#            raise NotImplementedError("HistogramObserver doesn't support dynamic quantization")
#        # bins: The number of bins used for histogram calculation.
#        super().__init__(
#            dtype=dtype,
#            qscheme=qscheme,
#            reduce_range=reduce_range,
#            quant_min=quant_min,
#            quant_max=quant_max,
#            factory_kwargs=factory_kwargs,
#            eps=eps,
#            is_dynamic=is_dynamic,
#            **kwargs,
#        )
#        factory_kwargs = torch.nn.factory_kwargs(factory_kwargs)
#        self.bins = bins
#        self.register_buffer("histogram", torch.zeros(self.bins, **factory_kwargs))
#        self.register_buffer("min_val", torch.tensor(float("inf"), **factory_kwargs))
#        self.register_buffer("max_val", torch.tensor(float("-inf"), **factory_kwargs))
#        self.dst_nbins = 2 ** torch.iinfo(self.dtype).bits
#        self.upsample_rate = 16  # used to reduce quantization errors when upscaling histogram
#
#    def _get_norm(self, delta_begin: torch.Tensor, delta_end: torch.Tensor, density: torch.Tensor) -> torch.Tensor:
#        r"""
#        Compute the norm of the values uniformaly distributed between
#        delta_begin and delta_end.
#        Currently only L2 norm is supported.
#
#        norm = density * (integral_{begin, end} x^2)
#             = density * (end^3 - begin^3) / 3
#        """
#        norm = (delta_end * delta_end * delta_end - delta_begin * delta_begin * delta_begin) / 3
#        return density * norm
#
#    def _compute_quantization_error(self, next_start_bin: int, next_end_bin: int):
#        r"""
#        Compute the quantization error if we use start_bin to end_bin as the
#        min and max to do the quantization.
#        """
#        bin_width = (self.max_val.item() - self.min_val.item()) / self.bins
#
#        dst_bin_width = bin_width * (next_end_bin - next_start_bin + 1) / self.dst_nbins
#        if dst_bin_width == 0.0:
#            return 0.0
#
#        src_bin = torch.arange(self.bins, device=self.histogram.device)
#        # distances from the beginning of first dst_bin to the beginning and
#        # end of src_bin
#        src_bin_begin = (src_bin - next_start_bin) * bin_width
#        src_bin_end = src_bin_begin + bin_width
#
#        # which dst_bins the beginning and end of src_bin belong to?
#        dst_bin_of_begin = torch.clamp(
#            torch.div(src_bin_begin, dst_bin_width, rounding_mode="floor"),
#            0,
#            self.dst_nbins - 1,
#        )
#        dst_bin_of_begin_center = (dst_bin_of_begin + 0.5) * dst_bin_width
#
#        dst_bin_of_end = torch.clamp(
#            torch.div(src_bin_end, dst_bin_width, rounding_mode="floor"),
#            0,
#            self.dst_nbins - 1,
#        )
#        density = self.histogram / bin_width
#
#        norm = torch.zeros(self.bins, device=self.histogram.device)
#
#        delta_begin = src_bin_begin - dst_bin_of_begin_center
#        delta_end = dst_bin_width / 2
#        norm += self._get_norm(
#            delta_begin,
#            torch.ones(self.bins, device=self.histogram.device) * delta_end,
#            density,
#        )
#
#        norm += (dst_bin_of_end - dst_bin_of_begin - 1) * self._get_norm(
#            torch.tensor(-dst_bin_width / 2), torch.tensor(dst_bin_width / 2), density
#        )
#
#        dst_bin_of_end_center = dst_bin_of_end * dst_bin_width + dst_bin_width / 2
#
#        delta_begin = -dst_bin_width / 2
#        delta_end = src_bin_end - dst_bin_of_end_center
#        norm += self._get_norm(torch.tensor(delta_begin), delta_end, density)
#
#        return norm.sum().item()
#
#    def _non_linear_param_search(self) -> tuple[torch.Tensor, torch.Tensor]:
#        r"""
#        Non-linear parameter search.
#
#        An approximation for L2 error minimization for selecting min/max.
#        By selecting new min/max, we filter out outliers in input distribution.
#        This follows the implementation of NormMinimization::NonlinearQuantizationParamsSearch in
#        caffe2/quantization/server/norm_minimization.cc
#        """
#        assert self.histogram.size()[0] == self.bins, "bins mismatch"
#        bin_width = (self.max_val - self.min_val) / self.bins
#
#        # cumulative sum
#        total = torch.sum(self.histogram).item()
#        cSum = torch.cumsum(self.histogram, dim=0)
#
#        stepsize = 1e-5  # granularity
#        alpha = 0.0  # lower bound
#        beta = 1.0  # upper bound
#        start_bin = 0
#        end_bin = self.bins - 1
#        norm_min = float("inf")
#
#        while alpha < beta:
#            # Find the next step
#            next_alpha = alpha + stepsize
#            next_beta = beta - stepsize
#
#            # find the left and right bins between the quantile bounds
#            l = start_bin
#            r = end_bin
#            while l < end_bin and cSum[l] < next_alpha * total:
#                l = l + 1
#            while r > start_bin and cSum[r] > next_beta * total:
#                r = r - 1
#
#            # decide the next move
#            next_start_bin = start_bin
#            next_end_bin = end_bin
#            if (l - start_bin) > (end_bin - r):
#                # move the start bin
#                next_start_bin = l
#                alpha = next_alpha
#            else:
#                # move the end bin
#                next_end_bin = r
#                beta = next_beta
#
#            if next_start_bin == start_bin and next_end_bin == end_bin:
#                continue
#
#            # calculate the quantization error using next_start_bin and next_end_bin
#            norm = self._compute_quantization_error(next_start_bin, next_end_bin)
#
#            if norm > norm_min:
#                break
#            norm_min = norm
#            start_bin = next_start_bin
#            end_bin = next_end_bin
#
#        new_min = self.min_val + bin_width * start_bin
#        new_max = self.min_val + bin_width * (end_bin + 1)
#        return new_min, new_max
#
#    def _upscale_histogram(
#        self,
#        histogram: torch.Tensor,
#        orig_min: torch.Tensor,
#        orig_max: torch.Tensor,
#        update_min: torch.Tensor,
#        update_max: torch.Tensor,
#    ):
#        # this turns the histogram into a more fine-coarsed histogram to reduce
#        # bin quantization errors
#        histogram = histogram.repeat_interleave(self.upsample_rate) / self.upsample_rate
#        bin_size = (orig_max - orig_min) / (self.bins * self.upsample_rate)
#        mid_points_histogram = (
#            torch.linspace(
#                orig_min,
#                orig_max,
#                self.bins * self.upsample_rate + 1,
#                device=orig_min.device,
#            )[:-1].to(histogram.device)
#            + 0.5 * bin_size
#        )
#        boundaries_new_histogram = torch.linspace(update_min, update_max, self.bins + 1, device=update_min.device).to(
#            histogram.device
#        )
#        # this maps the mid-poits of the histogram to the new histogram's space
#        bucket_assignments = torch.bucketize(mid_points_histogram, boundaries_new_histogram, right=True) - 1
#        # this then maps the histogram mid-points in the new space, weighted by the original histogram's values
#        # this is just the old histogram in the new histogram's space
#
#        # In case due to numerical issues the values land higher/lower than the maximum/minimum
#        bucket_assignments[bucket_assignments >= self.bins] = self.bins - 1
#        bucket_assignments[bucket_assignments < 0] = 0
#
#        update_histogram = torch.bincount(bucket_assignments, weights=histogram, minlength=self.bins)
#        return update_histogram
#
#    def _combine_histograms(
#        self,
#        orig_hist: torch.Tensor,
#        orig_min: torch.Tensor,
#        orig_max: torch.Tensor,
#        update_hist: torch.Tensor,
#        update_min: torch.Tensor,
#        update_max: torch.Tensor,
#    ) -> torch.Tensor:
#        # If the new min and max are the same as the current min and max,
#        # we can just add the new histogram to the original histogram
#        if update_min == orig_min and update_max == orig_max:
#            return orig_hist + update_hist
#
#        # If the orig hist only has one value (i.e., the min and max are the same)
#        # we can just add it into new histogram
#        if orig_min == orig_max:
#            bin_value = torch.sum(orig_hist)
#            transformed_orig_hist = (
#                torch.histc(orig_min, bins=self.bins, min=update_min, max=update_max)  # type: ignore[arg-type]
#                * bin_value
#            )
#            return transformed_orig_hist + update_hist
#
#        # We assume the update_hist is already in the target range, we will map the orig_max to it
#        assert update_min <= orig_min
#        assert update_max >= orig_max
#
#        # Now we need to turn the old_histogram, into the range of the new histogram
#        transformed_orig_hist = self._upscale_histogram(
#            orig_hist,
#            orig_min,
#            orig_max,
#            update_min,
#            update_max,
#        )
#
#        return update_hist + transformed_orig_hist
#
#    def reset_histogram(self, x: torch.Tensor, min_val: torch.Tensor, max_val: torch.Tensor) -> None:
#        self.min_val.resize_(min_val.shape)
#        self.min_val.copy_(min_val)
#        self.max_val.resize_(max_val.shape)
#        self.max_val.copy_(max_val)
#        assert min_val.numel() == 1 and max_val.numel() == 1, "histogram min/max values must be scalar."
#        new_histogram = torch.histc(x, self.bins, min=min_val, max=max_val)  # type: ignore[arg-type]
#        self.histogram.detach_().resize_(new_histogram.shape)
#        self.histogram.copy_(new_histogram)
#
#    def forward(self, x_orig: torch.Tensor) -> torch.Tensor:  # pyre-ignore[14]
#        if x_orig.numel() == 0:
#            return x_orig
#        x = x_orig.detach()
#        x_min, x_max = torch.aminmax(x)
#        # want to ignore torch.inf since we don't actually
#        # want to make our quantization range infinite
#        # and in practice those values will be clamped
#        if x_min == -torch.inf or x_max == torch.inf:
#            warnings.warn("torch.inf detected in input tensor, ignoring input")
#            x = x[x.abs() != torch.inf]
#            if x.numel() == 0:
#                return x_orig
#            x_min, x_max = torch.aminmax(x)
#
#        current_min = self.min_val
#        current_max = self.max_val
#
#        is_uninitialized = self.min_val == float("inf") or self.max_val == float("-inf")
#        if is_uninitialized:
#            self.reset_histogram(x, x_min, x_max)
#        else:
#            update_min, update_max = x_min, x_max
#            new_min = torch.min(current_min, update_min)
#            new_max = torch.max(current_max, update_max)
#
#            # TODO: For some reason, this is required for it to pass torchscript test
#            # new_min and new_max should already have requires_grad set to False
#            new_min, new_max = new_min.detach(), new_max.detach()
#            update_histogram = torch.histc(
#                x,
#                self.bins,
#                min=new_min,
#                max=new_max,  # type: ignore[arg-type]
#            ).to(self.histogram.device)
#            if new_min == current_min and new_max == current_max:
#                combined_histogram = self.histogram + update_histogram
#                self.histogram.detach_().resize_(combined_histogram.shape)
#                self.histogram.copy_(combined_histogram)
#            else:
#                combined_histogram = self._combine_histograms(
#                    self.histogram,
#                    current_min,
#                    current_max,
#                    update_histogram,
#                    new_min,
#                    new_max,
#                )
#                self.histogram.detach_().resize_(combined_histogram.shape)
#                self.histogram.copy_(combined_histogram)
#                self.min_val.detach_().resize_(new_min.shape)
#                self.min_val.copy_(new_min)
#                self.max_val.detach_().resize_(new_max.shape)
#                self.max_val.copy_(new_max)
#
#        return x_orig
#
#    @torch.jit.export
#    def calculate_qparams(self):
#        new_min, new_max = self._non_linear_param_search()
#        return new_min, new_max
#
#    def _save_to_state_dict(self, destination, prefix, keep_vars):
#        super()._save_to_state_dict(destination, prefix, keep_vars)
#        destination[prefix + "min_val"] = self.min_val
#        destination[prefix + "max_val"] = self.max_val
#
#    def _load_from_state_dict(
#        self,
#        state_dict,
#        prefix,
#        local_metadata,
#        strict,
#        missing_keys,
#        unexpected_keys,
#        error_msgs,
#    ):
#        version = local_metadata.get("version", None)
#
#        if version is None or version < 3:
#            # if min_val and max_val are not initialized, update their shape
#            # to account for the differences between v2 and v3
#            min_val_name, max_val_name = prefix + "min_val", prefix + "max_val"
#            if min_val_name in state_dict:
#                if state_dict[min_val_name].shape == torch.Size([0]):
#                    state_dict[min_val_name] = torch.tensor(float("inf"))
#            if max_val_name in state_dict:
#                if state_dict[max_val_name].shape == torch.Size([0]):
#                    state_dict[max_val_name] = torch.tensor(float("-inf"))
#
#        local_state = ["min_val", "max_val"]
#        for name in local_state:
#            key = prefix + name
#            if key in state_dict:
#                val = state_dict[key]
#                setattr(self, name, val)
#            elif strict:
#                missing_keys.append(key)
#        super()._load_from_state_dict(
#            state_dict,
#            prefix,
#            local_metadata,
#            strict,
#            missing_keys,
#            unexpected_keys,
#            error_msgs,
#        )
#
#    def extra_repr(self):
#        return f"min_val={self.min_val}, max_val={self.max_val}"
#
