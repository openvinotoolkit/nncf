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
import abc
from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from nncf.common.tensor_statistics.collectors import Tensor
from nncf.common.tensor_statistics.reduction import ReductionAxes
from nncf.common.tensor_statistics.reduction import is_reduce_to_scalar
from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.experimental.tensor import functions as fns
from nncf.quantization.advanced_parameters import AggregatorType

InplaceInsertionFNType = TypeVar("InplaceInsertionFNType")
AggregationAxes = Tuple[int, ...]


class TensorReducerBase(ABC):
    """
    Tensor reducer is a callable object that reduces tensors according to
    the specified rule. Could handle tensors inplace or out of place.
    """

    def __init__(
        self, reduction_axes: Optional[ReductionAxes] = None, channel_axis: Optional[int] = None, inplace: bool = False
    ):
        """
        :param reduction_axes: Reduction axes for reduction calculation. Equal to list(range(len(input.shape)))
            if empty.
        :param inplace: Whether should be calculated inplace or out of place.
        """
        if reduction_axes is None and channel_axis is None:
            raise RuntimeError("Either reduction_axes or channel_axis must be specified")

        if reduction_axes is not None and channel_axis is not None:
            raise RuntimeError("reduction_axes or channel_axis cannot be specified at the same time")
        self._reduction_axes = reduction_axes
        self._channel_axis = channel_axis
        self._inplace = inplace
        self._keepdims = True

    @property
    def inplace(self):
        return self._inplace

    @property
    def output_port_id(self) -> int:
        return 0

    @property
    def name(self):
        return self.__class__.__name__ + str(self.__hash__())

    @abstractmethod
    def _reduce_out_of_place(self, x: Tensor) -> Tensor:
        """
        Specifies the reduction rule in terms of NNCFTensor and NNCFTensorBackend opset.

        :param x: Tensor to register.
        """

    @abstractmethod
    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        """
        Returns target output names from target model that is
            modified for statistic collection.

        :param target_node_name: Target node name for reducer.
        :param port_id: Target port id for target node name for reducer.
        :return: Target output names for reducer.
        """

    @abstractmethod
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        """
        Returns correspondent inplace operation builder if inplace operations are available in backend.

        :return: Inplace operation builder if possible else None.
        """

    def __call__(self, x: List[Tensor]):
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

    def _get_axis(self, tensor: Tensor) -> Optional[ReductionAxes]:
        if self._reduction_axes is not None:
            if is_reduce_to_scalar(self._reduction_axes):
                return None
            return self._reduction_axes
        # self._channel_axis is not None
        proto_axis_list = list(range(tensor.ndim))
        del proto_axis_list[self._channel_axis]
        axis = tuple(proto_axis_list)
        return axis


class Aggregator(abc.ABC):
    def __init__(self, aggregation_axes: Optional[AggregationAxes] = None, num_samples: Optional[int] = None):
        """
        :param num_samples: Maximum number of samples to collect. Aggregator
            skips tensor registration if tensor registration was called num_samples times before.
            Aggregator never skips registration if num_samples is None.
        :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
        """

        self._num_samples = num_samples
        self._collected_samples = 0
        self._keepdims = True
        self._aggregation_axes = (0,) if aggregation_axes is None else aggregation_axes

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def register_reduced_input(self, x: Tensor):
        if self._num_samples is not None and self._collected_samples >= self._num_samples:
            return
        self._register_reduced_input_impl(x)
        self._collected_samples += 1

    @abstractmethod
    def _register_reduced_input_impl(self, x: Tensor) -> None:
        """
        Registers incoming tensor in tensor aggregator.

        :param x: Tensor to register.
        """

    def aggregate(self) -> Optional[Tensor]:
        """
        Aggregates collected tensors and returns aggregated result.
        In case no tensors were collected returns None.

        :return: Aggregated result.
        """
        if self._collected_samples:
            return self._aggregate_impl()
        return None

    @abstractmethod
    def _aggregate_impl(self) -> Tensor:
        """
        Aggregates collected tensors and returns aggregated result.

        :return: Aggregated result.
        """

    def reset(self):
        self._collected_samples = 0
        self._reset_sample_container()

    @abstractmethod
    def _reset_sample_container(self):
        pass

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and self._num_samples == __o.num_samples

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


class AggregatorBase(Aggregator, abc.ABC):
    """
    Tensor aggregator is designed to receive (register) calculated statistics and
    aggregate them in terms of NNCFTensor and NNCFTensorBackend opset.
    """

    def aggregate(self) -> Optional[Tensor]:
        """
        Aggregates collected tensors and returns aggregated result.
        In case no tensors were collected returns None.

        :return: Aggregated result.
        """
        if self._collected_samples:
            return self._aggregate_impl()
        return None


class OnlineTensorAggregator(AggregatorBase, abc.ABC):
    def __init__(self, aggregation_axes: AggregationAxes, num_samples: Optional[int] = None):
        super().__init__(aggregation_axes=aggregation_axes, num_samples=num_samples)
        self._current_aggregate = None

    def _reset_sample_container(self):
        self._current_aggregate = None

    def _online_aggregation_axes(self) -> AggregationAxes:
        return tuple(dim - 1 for dim in self._aggregation_axes if dim != 0)


AggregatorKey = Tuple[int, int]


class TensorCollector:
    """
    Calculates statistics at given tensors according to registered statistic branches.
    Statistic branch consists of one reducer and one aggregator instance. TensorCollector
    applies a reducer on a correspondent inputs and then passes the one of the reduced tensors
    chosen by output port id to a correspondent aggregator for each registered statistic branch.
    Receives tensors by `register_input` method. Aggregated values as a TensorStatistic instance or
    a dict could be collected by `get_statistics` call.
    """

    def __init__(self, statistic_container: Optional[Type[TensorStatistic]] = None) -> None:
        self._reducers: Set[TensorReducerBase] = set()
        self._aggregators: Dict[AggregatorKey, AggregatorBase] = {}
        self._stat_container_kwargs_map: Dict[str, Tuple[int, int]] = {}
        self._stat_container = statistic_container
        self._enabled = True

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
            raise RuntimeError(
                f"Two different statistic branches for one container key {container_key} are encountered"
            )
        if any(aggr is aggregator for aggr in self._aggregators.values()):
            raise RuntimeError(f"One aggregator instance {aggregator} for different branches is encountered")

        self._reducers.add(reducer)
        key = (hash(reducer), reducer_output_port_id, hash(aggregator))

        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        self._stat_container_kwargs_map[container_key] = key

    def get_output_info(self, target_node_name: str, port_id: int) -> List[Tuple[int, List[str]]]:
        """
        Returns list of pairs of reducers names and correspondent output names.

        :param target_node_name: Target node name to assemble output name.
        :param port_id: Target node specific port id to assemble output name.
        :returns: List of pairs of reducers hashes and correspondent output names.
        """
        retval = []
        for reducer in self._reducers:
            retval.append((hash(reducer), reducer.get_output_names(target_node_name, port_id)))
        return retval

    def register_inputs(self, inputs: Dict[int, List[Tensor]]) -> None:
        """
        Registers given input in TensorCollector.

        :param inputs: Tensor inputs in format of dict where keys
            are reducer names and values are correspondent input tensors
        """
        if not self._enabled:
            return

        reduced_inputs = {}
        for reducer in self._reducers:
            reducer_hash = hash(reducer)
            input_ = inputs[reducer_hash]
            if any(tensor.is_empty() for tensor in input_):
                continue
            reduced_inputs[reducer_hash] = reducer(input_)

        for (
            (reducer_hash, reducer_port_id, _),
            aggregator,
        ) in self._aggregators.items():
            if reducer_hash in reduced_inputs:
                aggregator.register_reduced_input(reduced_inputs[reducer_hash][reducer_port_id])

    def register_input_for_all_reducers(self, input_: Tensor) -> None:
        """
        Registers given input_ in each avaliable statistic collection branch.

        :param input_: Tensor input to register.
        """
        self.register_inputs({hash(reducer): [input_] for reducer in self._reducers})

    def _aggregate(self) -> Dict[AggregatorKey, Optional[Tensor]]:
        result = {}
        for (
            key,
            aggregator,
        ) in self._aggregators.items():
            val = aggregator.aggregate()
            result[key] = val
        return result

    def get_statistics(self) -> Union[TensorStatistic, Dict[str, Any]]:
        """
        Returns aggregated values in format of a TensorStatistic instance or
        a dict.

        :returns: Aggregated values.
        """

        aggregated_values = self._aggregate()
        kwargs = {}
        for container_key, branch_key in self._stat_container_kwargs_map.items():
            kwargs[container_key] = aggregated_values[branch_key]

        if not self._stat_container:
            return kwargs
        return self._build_statistic_container(self._stat_container, kwargs)

    def get_inplace_fn_info(self) -> List[Tuple[Any, int]]:
        """
        Returns necessary information to insert inplace operation into graph.

        :returns: necessary information to insert inplace operation into graph
            in format of pair of reducer builder and correspondent reducer output port id.
        """
        retval = []
        for reducer in self._reducers:
            if reducer.inplace:
                retval.append((reducer.get_inplace_fn(), reducer.output_port_id))
        return retval

    def any_stat_out_of_place(self) -> bool:
        """
        Returns True if any reducer is calculated out of place.

        :returns: True if any reducer is calculated out of place.
        """
        return any(not reducer.inplace for reducer in self._reducers)

    def replace_aggregator(self, key: Tuple[int, int, int], aggregator: AggregatorBase) -> None:
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
        outputs: Dict[str, Tensor], output_info: List[Tuple[int, List[str]]]
    ) -> Dict[int, List[Tensor]]:
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

    @staticmethod
    def _build_statistic_container(statistic_container_cls: Type[TensorStatistic], kwargs: Dict[Any, Any]):
        if issubclass(statistic_container_cls, MinMaxTensorStatistic):
            return statistic_container_cls(
                min_values=kwargs[MinMaxTensorStatistic.MIN_STAT], max_values=kwargs[MinMaxTensorStatistic.MAX_STAT]
            )
        if issubclass(statistic_container_cls, MeanTensorStatistic):
            return statistic_container_cls(
                mean_values=kwargs[MeanTensorStatistic.MEAN_STAT], shape=kwargs[MeanTensorStatistic.SHAPE_STAT]
            )
        if issubclass(statistic_container_cls, RawTensorStatistic):
            return statistic_container_cls(values=kwargs[RawTensorStatistic.VALUES_STATS])
        if issubclass(statistic_container_cls, MedianMADTensorStatistic):
            return statistic_container_cls(
                median_values=kwargs[MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY][
                    MedianMADTensorStatistic.MEDIAN_VALUES_STAT
                ],
                mad_values=kwargs[MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY][
                    MedianMADTensorStatistic.MAD_VALUES_STAT
                ],
            )
        if issubclass(statistic_container_cls, PercentileTensorStatistic):
            if PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY in kwargs:
                percentile_vs_values_dict = kwargs[PercentileTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY]
            else:
                percentile_vs_values_dict = {}
                for (_, percentile), value in kwargs.items():
                    percentile_vs_values_dict[percentile] = value
            return statistic_container_cls(percentile_vs_values_dict=percentile_vs_values_dict)
        raise RuntimeError(
            f"Statistic collector class {statistic_container_cls} is not supported by the TensorCollector class."
        )


class MergedTensorCollector(TensorCollector):
    """
    Tensor collector that merge several tensor collectors in one.
    Statistics collected by a merged tensor collector automatically available
    in all tensor collectors that were merged by the merged tensor collector.
    This works because merged tensor collectors share tensor aggregators instances with
    the merged tensor collector.
    """

    def __init__(self, tensor_collectors: List[TensorCollector]) -> None:
        """
        :param tensor_collectors: Tensor collectors to merge.
        """
        super().__init__()
        aggregators: Dict[Tuple[int, int, int], List[Tuple[TensorCollector, AggregatorBase]]] = defaultdict(list)
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


##################################################Reducers##################################################


class NoopReducer(TensorReducerBase):
    def __init__(self):
        super().__init__(reduction_axes=tuple(), inplace=False)

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        return x


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]
        axis = self._get_axis(x)
        return [fns.amin(x, axis=axis, keepdims=self._keepdims)]


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]
        axis = self._get_axis(x)

        return [fns.amax(x, axis=axis, keepdims=self._keepdims)]


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]

        x = fns.abs(x)
        axis = self._get_axis(x)
        return [fns.amax(x, axis=axis, keepdims=self._keepdims)]


class MeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]

        axis = self._get_axis(x)
        return [fns.mean(x, axis, keepdims=self._keepdims)]


class QuantileReducerBase(TensorReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        channel_axis: Optional[int] = None,
        quantile: Optional[Union[float, Tuple[float]]] = None,
        inplace: bool = False,
    ):
        super().__init__(reduction_axes=reduction_axes, channel_axis=channel_axis, inplace=False)
        self._quantile = (0.01, 0.99) if quantile is None else quantile

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, tuple(self._quantile)))


class QuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]
        axis = self._get_axis(x)

        return list(fns.quantile(x, self._quantile, axis=axis, keepdims=self._keepdims))


class AbsQuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]

        x = fns.abs(x)
        axis = self._get_axis(x)
        return [fns.quantile(x, self._quantile, axis=axis, keepdims=self._keepdims)]


class BatchMeanReducer(TensorReducerBase):
    def __init__(self, inplace: bool = False):
        super().__init__(channel_axis=0, inplace=inplace)

    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]

        return [fns.mean(x, axis=0, keepdims=True)]


class MeanPerChReducer(TensorReducerBase):
    def __init__(self, channel_axis: int = 1, inplace: bool = False):
        super().__init__(channel_axis=channel_axis, inplace=inplace)

    def _reduce_out_of_place(self, x: List[Tensor]) -> List[Tensor]:
        x = x[0]

        axis = self._get_axis(x)
        retval = fns.mean(x, axis=axis, keepdims=True)
        return [retval]


##################################################Aggregators##################################################


class NoopAggregator(AggregatorBase):
    # TODO (vshampor): this seems to be a bad design since this class does not actually aggregate anything
    #  and therefore cannot be an `Aggregator` instance. In particular this is visible in the interface violation
    #  for the _aggregate_impl function, which is supposed to return a single tensor - the aggregate - but has to
    #  return a list of tensors instead in order for the purpose of the class to be served.
    def _reset_sample_container(self):
        pass

    def __init__(self, num_samples: Optional[int]):
        super().__init__(num_samples=num_samples)
        self._unaggregated_samples = []

    def _register_reduced_input_impl(self, x: Tensor) -> None:
        self._unaggregated_samples.append(x)

    def _aggregate_impl(self) -> List[Tensor]:
        return self._unaggregated_samples


class ShapeAggregator(AggregatorBase):
    def __init__(self):
        super().__init__(num_samples=1)
        self._shape = None

    def _register_reduced_input_impl(self, x: Tensor) -> None:
        self._shape = x.shape

    def _aggregate_impl(self) -> List[int]:
        return self._shape

    def _reset_sample_container(self):
        self._shape = None


class MinAggregator(OnlineTensorAggregator):
    def _register_reduced_input_impl(self, x: Tensor) -> None:
        if self._current_aggregate is None:
            self._current_aggregate = x
        else:
            self._current_aggregate = fns.minimum(x, self._current_aggregate)

    def _aggregate_impl(self) -> Tensor:
        return self._current_aggregate


class MaxAggregator(OnlineTensorAggregator):
    def _register_reduced_input_impl(self, x: Tensor) -> None:
        if self._current_aggregate is None:
            self._current_aggregate = x
        else:
            self._current_aggregate = fns.maximum(x, self._current_aggregate)

    def _aggregate_impl(self) -> Tensor:
        return self._current_aggregate


class OfflineAggregatorBase(AggregatorBase, ABC):
    def __init__(
        self,
        aggregation_axes: Optional[AggregationAxes] = None,
        use_per_sample_stats: bool = False,
        num_samples: Optional[int] = None,
        window_size: int = None,
    ):
        super().__init__(aggregation_axes=aggregation_axes, num_samples=num_samples)
        self._window_size = window_size
        self._samples: Deque[Tensor] = deque(maxlen=window_size)
        self._use_per_sample_stats = use_per_sample_stats

    def _register_reduced_input_impl(self, x: Tensor) -> None:
        if self._use_per_sample_stats:
            self._samples.extend(fns.unstack(x))
        else:
            self._samples.append(x)

    def _aggregate_impl(self) -> Tensor:
        stacked_val = fns.stack(list(self._samples))
        aggregated = self._aggregate_stacked_samples(stacked_val)
        return aggregated.squeeze(0)

    def _reset_sample_container(self):
        self._samples.clear()

    @abstractmethod
    def _aggregate_stacked_samples(self, stacked_samples: Tensor) -> Tensor:
        pass


class MeanAggregator(OfflineAggregatorBase):
    def _aggregate_stacked_samples(self, stacked_samples: Tensor) -> Tensor:
        return stacked_samples.mean(axis=self._aggregation_axes, keepdims=self._keepdims)


class MedianAggregator(OfflineAggregatorBase):
    def _aggregate_stacked_samples(self, stacked_samples: Tensor) -> Tensor:
        return stacked_samples.median(axis=self._aggregation_axes, keepdims=self._keepdims)


class NoOutliersAggregatorBase(OfflineAggregatorBase, ABC):
    def __init__(
        self,
        aggregation_axes: AggregationAxes,
        use_per_sample_stats: bool = False,
        num_samples: Optional[int] = None,
        window_size: int = None,
        quantile: float = 0.01,
    ):
        super().__init__(
            aggregation_axes=aggregation_axes,
            use_per_sample_stats=use_per_sample_stats,
            num_samples=num_samples,
            window_size=window_size,
        )
        self._quantile = quantile

    def _aggregate_stacked_samples(self, stacked_samples: Tensor) -> Tensor:
        alpha = self._quantile

        low_values, high_values = fns.quantile(stacked_samples, [alpha, 1 - alpha], 0)
        outliers_mask = fns.logical_or(stacked_samples < low_values, high_values < stacked_samples)
        return self._aggregate_stacked_samples_with_no_outliers(stacked_samples, outliers_mask)

    @abstractmethod
    def _aggregate_stacked_samples_with_no_outliers(self, stacked_val: Tensor, outliers_mask: Tensor) -> Tensor:
        pass

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._quantile))


class MeanNoOutliersAggregator(NoOutliersAggregatorBase):
    def _aggregate_stacked_samples_with_no_outliers(self, stacked_val: Tensor, outliers_mask: Tensor) -> Tensor:
        return fns.masked_mean(stacked_val, mask=outliers_mask, axis=self._aggregation_axes)


class MedianNoOutliersAggregator(NoOutliersAggregatorBase):
    def _aggregate_stacked_samples_with_no_outliers(self, stacked_val: Tensor, outliers_mask: Tensor) -> Tensor:
        return fns.masked_median(stacked_val, mask=outliers_mask, axis=self._aggregation_axes)


class MedianAbsoluteDeviationAggregator(OfflineAggregatorBase):
    def _aggregate_stacked_samples(self, stacked_samples: Tensor) -> Tensor:
        pass

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
            raise NotImplementedError(
                "Aggregation without 0 dim is not supported yet for MedianAbsoluteDeviationAggregator"
            )

    def _register_reduced_input_impl(self, x: Tensor) -> None:
        return self._samples.append(x)



    def _aggregate_impl(self) -> Dict[str, Tensor]:
        stacked_val, shape_after_aggregation = _moveaxes_flatten_cat(
            list(self._samples), [x - 1 for x in self._aggregation_axes if x > 0]
        )

        mask = fns.abs(stacked_val) < fns.eps(stacked_val)  # zeros mask

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


class PercentileAggregator(OfflineAggregatorBase):
    def __init__(
        self,
        percentiles_to_collect: List[float],
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
    ):
        super().__init__(aggregation_axes=aggregation_axes, num_samples=num_samples)
        if 0 not in self._aggregation_axes:
            raise NotImplementedError("Aggregation without 0 dim is not supported yet for PercentileAggregator")
        self._quantiles_to_collect = [x / 100 for x in percentiles_to_collect]

    def _register_reduced_input_impl(self, x: Tensor) -> None:
        return self._samples.append(x)

    def _aggregate_impl(self) -> Dict[float, Tensor]:
        stacked_val, shape_after_aggregation = _moveaxes_flatten_cat(
            list(self._samples), [x - 1 for x in self._aggregation_axes if x > 0]
        )

        percentiles = fns.quantile(stacked_val, self._quantiles_to_collect, axis=0, keepdims=False)
        retval = {}
        for idx, percentile in enumerate(self._quantiles_to_collect):
            value = percentiles[idx]
            if self._keepdims:
                value = fns.reshape(value, shape_after_aggregation)
            retval[percentile] = value
        return retval


def _moveaxes_flatten_cat(
    tensor_list: List[Tensor], aggregation_axes: AggregationAxes
) -> Tuple[Tensor, Tuple[int, ...]]:
    """
    Moves aggregation axes to the begining of the tensor shape for each tensor from the list, flattens
    and concatenates them in 0 dimension. Computes target shape for the processed tensor
    after an aggregation function is applied to it. Target shape preserves original order
    of dimensions and replaces aggregated dimensions by 1.

    :param tensor_list: Tensor list to process.
    :param aggregation_axes: Aggregation axes to move, flatten and concatinate.
    :return: Tuple of the processed tensor and
        target shape for the processed tensor after an aggregation function is applied to it.
    """
    tensor_shape = list(tensor_list[0].shape)

    # Transpose dims to move aggregation axes forward
    transpose_dims = list(range(len(tensor_shape)))
    for idx, axis in enumerate(aggregation_axes):
        transpose_dims[axis], transpose_dims[idx] = transpose_dims[idx], transpose_dims[axis]

    # Shape to flatten aggregation axes
    reshape_shape = [
        -1,
    ] + [
        tensor_shape[dim] for dim in transpose_dims
    ][len(aggregation_axes) :]

    reshaped_tensors = []
    for tensor in tensor_list:
        transposed_t = fns.transpose(tensor, transpose_dims)
        reshaped_tensors.append(fns.reshape(transposed_t, reshape_shape))

    shape_after_aggregation = tuple(1 if idx in aggregation_axes else dim for idx, dim in enumerate(tensor_shape))
    return fns.concatenate(reshaped_tensors, axis=0), shape_after_aggregation


AGGREGATORS_MAP = {
    AggregatorType.MIN: MinAggregator,
    AggregatorType.MAX: MaxAggregator,
    AggregatorType.MEAN: MeanAggregator,
    AggregatorType.MEAN_NO_OUTLIERS: MeanNoOutliersAggregator,
    AggregatorType.MEDIAN: MedianAggregator,
    AggregatorType.MEDIAN_NO_OUTLIERS: MedianNoOutliersAggregator,
}
