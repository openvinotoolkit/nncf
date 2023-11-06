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
from collections import defaultdict
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from nncf.common.tensor import TensorType
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import NNCFTensor
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.quantization.advanced_parameters import AggregatorType

InplaceInsertionFNType = TypeVar("InplaceInsertionFNType")
AggregationAxes = Tuple[int, ...]


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
        self._tensor_processor: NNCFCollectorTensorProcessor = self._get_processor()
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

    @staticmethod
    @abstractmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        pass

    @abstractmethod
    def _reduce_out_of_place(self, x: List[TensorType]) -> List[TensorType]:
        """
        Specifies the reduction rule in terms of NNCFCollectorTensorProcessor.

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

    def __call__(self, x: List[NNCFTensor]):
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

    def _get_reduction_axes(self, tensor: NNCFTensor) -> ReductionAxes:
        if self._reduction_axes is not None:
            return self._reduction_axes
        return tuple(range(len(tensor.shape)))


class AggregatorBase:
    """
    Aggregator is designed to receive (register) calculated statistics and
    aggregate them in terms of NNCFCollectorTensorProcessor operations.
    """

    def __init__(
        self,
        tensor_processor: NNCFCollectorTensorProcessor,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size: Optional[int] = None,
    ):
        """
        :param tensor_processor: Backend-specific tensor processor.
        :param aggregation_axes: Axes along which to operate.
            Registered statistics are stacked along zero axis,
            axes >=1 correspond to recieved statistic axes shifted left by 1.
        :param num_samples: Maximum number of samples to collect. Aggregator
            skips tensor registration if tensor registration was called num_samples times before.
            Aggregator never skips registration if num_samples is None.
        :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
        """

        self._tensor_processor = tensor_processor
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

    def __init__(self, statistic_container: Optional[TensorStatistic] = None) -> None:
        self._reducers: Set[TensorReducerBase] = set()
        self._aggregators: Dict[Tuple[int, int, int], AggregatorBase] = {}
        self._stat_container_kwargs_map: Dict[str, Tuple[int, int, int]] = {}
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

    def register_inputs(self, inputs: Dict[int, List[NNCFTensor]]) -> None:
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

    def register_input_for_all_reducers(self, input_: NNCFTensor) -> None:
        """
        Registers given input_ in each avaliable statistic collection branch.

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
        outputs: Dict[str, NNCFTensor], output_info: List[Tuple[int, List[str]]]
    ) -> Dict[int, List[NNCFTensor]]:
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
        super().__init__(inplace=False)

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return None

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def _reduce_out_of_place(self, x: List[TensorType]) -> List[TensorType]:
        return x


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.reduce_min(x, reduction_axes, keepdims=self._keepdims)]


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.reduce_max(x, reduction_axes, keepdims=self._keepdims)]


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = self._tensor_processor.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.reduce_max(x, reduction_axes, keepdims=self._keepdims)]


class MeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return [self._tensor_processor.mean(x, reduction_axes, keepdims=self._keepdims)]


class QuantileReducerBase(TensorReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        quantile: Optional[Union[float, Tuple[float]]] = None,
        inplace: bool = False,
    ):
        super().__init__(reduction_axes=reduction_axes, inplace=False)
        self._quantile = (0.01, 0.99) if quantile is None else quantile

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, tuple(self._quantile)))


class QuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        reduction_axes = self._get_reduction_axes(x)
        return self._tensor_processor.quantile(x, self._quantile, reduction_axes, keepdims=self._keepdims)


class AbsQuantileReducer(QuantileReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        quantile: Optional[Union[float, List[float]]] = None,
        inplace: bool = False,
    ):
        quantile = (0.99,) if quantile is None else quantile
        super().__init__(reduction_axes=reduction_axes, quantile=quantile, inplace=False)

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = self._tensor_processor.abs(x[0])
        reduction_axes = self._get_reduction_axes(x)
        return self._tensor_processor.quantile(x, self._quantile, reduction_axes, keepdims=self._keepdims)


class BatchMeanReducer(TensorReducerBase):
    def __init__(self, inplace: bool = False):
        super().__init__(None, inplace)

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        return [self._tensor_processor.batch_mean(x[0])]


class MeanPerChReducer(TensorReducerBase):
    def __init__(self, channel_axis: int = 1, inplace: bool = False):
        super().__init__(inplace=inplace)
        self._channel_axis = channel_axis

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        return [self._tensor_processor.mean_per_channel(x[0], self._channel_axis)]

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._channel_axis == __o._channel_axis

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, self._channel_axis))


##################################################Aggregators##################################################


class NoopAggregator(AggregatorBase):
    def __init__(self, num_samples: Optional[int]):
        super().__init__(None, num_samples=num_samples)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container.append(x.tensor)

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

    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        online_aggregation_axes = tuple(dim - 1 for dim in self._aggregation_axes if dim != 0)
        if online_aggregation_axes:
            reduced = self._aggregation_fn(x, axis=online_aggregation_axes, keepdims=self._keepdims)
        else:
            reduced = x
        if 0 in self._aggregation_axes:
            stacked_tensors = self._tensor_processor.stack([reduced, *self._container], axis=0)
            aggregated = self._aggregation_fn(stacked_tensors, axis=0, keepdims=self._keepdims)
            aggregated = self._tensor_processor.squeeze(aggregated, 0)
            self._container = [aggregated]
        else:
            self._container.append(reduced)

    def _aggregate_impl(self) -> NNCFTensor:
        if 0 in self._aggregation_axes:
            if self._keepdims:
                return self._container[0].tensor
        return self._tensor_processor.stack(self._container).tensor

    @abstractmethod
    def _aggregation_fn(self, stacked_value: NNCFTensor, axis: AggregationAxes, keepdims: bool) -> NNCFTensor:
        pass


class MinAggregator(OnlineAggregatorBase):
    def _aggregation_fn(self, stacked_value: NNCFTensor, axis: AggregationAxes, keepdims: bool) -> NNCFTensor:
        return self._tensor_processor.reduce_min(stacked_value, axis=axis, keepdims=keepdims)


class MaxAggregator(OnlineAggregatorBase):
    def _aggregation_fn(self, stacked_value: NNCFTensor, axis: AggregationAxes, keepdims: bool) -> NNCFTensor:
        return self._tensor_processor.reduce_max(stacked_value, axis=axis, keepdims=keepdims)


class OfflineAggregatorBase(AggregatorBase, ABC):
    """
    Base class for aggregators which are using aggregation function fn which
    does not fulfill property fn([x1, x2, x3]) == fn([fn([x1, x2]), x3])
    where x1, x2, x3 are samples to aggregate. Child aggregators collect
    all samples in a container and aggregate them in one step.
    """

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        self._container.append(x)

    def _aggregate_impl(self) -> NNCFTensor:
        stacked_val = self._tensor_processor.stack(self._container)
        aggregated = self._aggregation_fn(stacked_val, axis=self._aggregation_axes, keepdims=self._keepdims)
        return self._tensor_processor.squeeze(aggregated, 0).tensor

    @abstractmethod
    def _aggregation_fn(self, stacked_value: NNCFTensor, axis: AggregationAxes, keepdims: bool) -> NNCFTensor:
        pass


class MeanAggregator(OfflineAggregatorBase):
    def _aggregation_fn(self, stacked_value: NNCFTensor, axis: AggregationAxes, keepdims: bool) -> NNCFTensor:
        return self._tensor_processor.mean(stacked_value, axis=axis, keepdims=keepdims)


class MedianAggregator(OfflineAggregatorBase):
    def _aggregation_fn(self, stacked_value: NNCFTensor, axis: AggregationAxes, keepdims: bool) -> NNCFTensor:
        return self._tensor_processor.median(stacked_value, axis=axis, keepdims=keepdims)


class NoOutliersAggregatorBase(OfflineAggregatorBase, ABC):
    def __init__(
        self,
        tensor_processor: NNCFCollectorTensorProcessor,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
        quantile: float = 0.01,
    ):
        super().__init__(tensor_processor, aggregation_axes=aggregation_axes, num_samples=num_samples)
        self._window_size = window_size
        self._container = deque(maxlen=window_size)
        self._quantile = quantile

    def _aggregate_impl(self) -> NNCFTensor:
        stacked_samples = self._tensor_processor.stack(self._container)
        low_values, high_values = self._tensor_processor.quantile(
            stacked_samples,
            quantile=(self._quantile, 1 - self._quantile),
            axis=self._aggregation_axes,
        )
        tp = self._tensor_processor
        outliers_mask = tp.logical_or(tp.less(stacked_samples, low_values), tp.less(high_values, stacked_samples))
        aggregated = self._aggregation_fn(
            stacked_samples=stacked_samples,
            mask=outliers_mask,
            axis=self._aggregation_axes,
            keepdims=self._keepdims,
        )
        return self._tensor_processor.squeeze(aggregated, 0).tensor

    @abstractmethod
    def _aggregation_fn(
        self, stacked_samples: NNCFTensor, mask: NNCFTensor, axis: AggregationAxes, keepdims: bool
    ) -> NNCFTensor:
        pass

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._quantile))


class MeanNoOutliersAggregator(NoOutliersAggregatorBase):
    def _aggregation_fn(
        self, stacked_samples: NNCFTensor, mask: NNCFTensor, axis: AggregationAxes, keepdims: bool
    ) -> NNCFTensor:
        return self._tensor_processor.masked_mean(stacked_samples, axis=axis, mask=mask, keepdims=keepdims)


class MedianNoOutliersAggregator(NoOutliersAggregatorBase):
    def _aggregation_fn(
        self, stacked_samples: NNCFTensor, mask: NNCFTensor, axis: AggregationAxes, keepdims: bool
    ) -> NNCFTensor:
        return self._tensor_processor.masked_median(stacked_samples, axis=axis, mask=mask, keepdims=keepdims)


class MedianAbsoluteDeviationAggregator(AggregatorBase):
    def __init__(
        self,
        tensor_processor: NNCFCollectorTensorProcessor,
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
    ):
        super().__init__(
            tensor_processor=tensor_processor,
            aggregation_axes=aggregation_axes,
            num_samples=num_samples,
            window_size=window_size,
        )
        if 0 not in self._aggregation_axes:
            raise NotImplementedError(
                "Aggregation without 0 dim is not supported yet for MedianAbsoluteDeviationAggregator"
            )

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        return self._container.append(x)

    def _aggregate_impl(self) -> Dict[str, NNCFTensor]:
        stacked_val, shape_after_aggregation = _moveaxes_flatten_cat(
            self._container, [x - 1 for x in self._aggregation_axes if x > 0], self._tensor_processor
        )

        mask = self._tensor_processor.zero_elements(stacked_val)
        median_per_ch = self._tensor_processor.masked_median(stacked_val, mask=mask, axis=0, keepdims=True)

        mad_values = self._tensor_processor.median(
            self._tensor_processor.abs(self._tensor_processor.sub(stacked_val, median_per_ch)),
            axis=0,
            keepdims=False,
        )
        if self._keepdims:
            median_per_ch = self._tensor_processor.reshape(median_per_ch, shape_after_aggregation)
            mad_values = self._tensor_processor.reshape(mad_values, shape_after_aggregation)
        else:
            median_per_ch = self._tensor_processor.squeeze(median_per_ch, 0)
        return {
            MedianMADTensorStatistic.MEDIAN_VALUES_STAT: median_per_ch.tensor,
            MedianMADTensorStatistic.MAD_VALUES_STAT: mad_values.tensor,
        }


class PercentileAggregator(AggregatorBase):
    def __init__(
        self,
        tensor_processor: NNCFCollectorTensorProcessor,
        percentiles_to_collect: List[float],
        aggregation_axes: Optional[AggregationAxes] = None,
        num_samples: Optional[int] = None,
        window_size=None,
    ):
        super().__init__(tensor_processor, aggregation_axes=aggregation_axes, num_samples=num_samples)
        if 0 not in self._aggregation_axes:
            raise NotImplementedError("Aggregation without 0 dim is not supported yet for PercentileAggregator")
        self._percentiles_to_collect = percentiles_to_collect
        self._window_size = window_size
        self._container = deque(maxlen=window_size)

    def _register_reduced_input_impl(self, x: TensorType) -> None:
        return self._container.append(x)

    def _aggregate_impl(self) -> Dict[float, NNCFTensor]:
        stacked_val, shape_after_aggregation = _moveaxes_flatten_cat(
            self._container, [x - 1 for x in self._aggregation_axes if x > 0], self._tensor_processor
        )

        percentiles = self._tensor_processor.percentile(
            stacked_val, self._percentiles_to_collect, axis=0, keepdims=False
        )
        retval = {}
        for idx, percentile in enumerate(self._percentiles_to_collect):
            value = percentiles[idx]
            if self._keepdims:
                value = self._tensor_processor.reshape(value, shape_after_aggregation)
            retval[percentile] = value.tensor
        return retval


def _moveaxes_flatten_cat(
    tensor_list: List[NNCFTensor], aggregation_axes: Tuple[int, ...], tensor_processor: NNCFCollectorTensorProcessor
) -> Tuple[NNCFTensor, Tuple[int, ...]]:
    """
    Moves aggregation axes to the begining of the tensor shape for each tensor from the list, flattens
    and concatenates them in 0 dimension. Computes target shape for the processed tensor
    after an aggregation function is applied to it. Target shape preserves original order
    of dimensions and replaces aggregated dimensions by 1.

    :param tensor_list: NNCFTensor list to process.
    :param aggregation_axes: Aggregation axes to move, flatten and concatinate.
    :param tensor_processor: Backed-specific tensor processor instance.
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
        transposed_t = tensor_processor.transpose(tensor, transpose_dims)
        reshaped_tensors.append(tensor_processor.reshape(transposed_t, reshape_shape))

    shape_after_aggregation = tuple(1 if idx in aggregation_axes else dim for idx, dim in enumerate(tensor_shape))
    return tensor_processor.cat(reshaped_tensors, axis=0), shape_after_aggregation


AGGREGATORS_MAP = {
    AggregatorType.MIN: MinAggregator,
    AggregatorType.MAX: MaxAggregator,
    AggregatorType.MEAN: MeanAggregator,
    AggregatorType.MEAN_NO_OUTLIERS: MeanNoOutliersAggregator,
    AggregatorType.MEDIAN: MedianAggregator,
    AggregatorType.MEDIAN_NO_OUTLIERS: MedianNoOutliersAggregator,
}
