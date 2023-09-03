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
from typing import Any, Deque, Dict, List, Optional, Set, Tuple, TypeVar, Union
from typing import Type

from nncf.common.tensor_statistics.collectors import NNCFTensor
from nncf.common.tensor_statistics.collectors import ReductionAxes
from nncf.common.tensor_statistics.collectors import is_reduce_to_scalar
from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.quantization.advanced_parameters import AggregatorType

InplaceInsertionFNType = TypeVar("InplaceInsertionFNType")


class TensorReducerBase(ABC):
    """
    Tensor reducer is a callable object that reduces tensors according to
    the specified rule. Could handle tensors inplace or out of place.
    """

    def __init__(self, reduction_axes: Optional[ReductionAxes] = None, channel_axis: Optional[int] = None, inplace: bool = False):
        """
        :param reduction_axes: Reduction shape for reduction calculation. Equal to list(range(len(input.shape)))
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
    def _reduce_out_of_place(self, x: NNCFTensor) -> NNCFTensor:
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

    def _get_axis(self, tensor: NNCFTensor) -> Optional[ReductionAxes]:
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
    def __init__(self, num_samples: Optional[int] = None):
        """
        :param num_samples: Maximum number of samples to collect. Aggregator
            skips tensor registration if tensor registration was called num_samples times before.
            Aggregator never skips registration if num_samples is None.
        """

        self._num_samples = num_samples
        self._collected_samples = 0

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def register_reduced_input(self, x: NNCFTensor):
        if self._num_samples is not None and self._collected_samples >= self._num_samples:
            return
        self._register_reduced_input_impl(x)
        self._collected_samples += 1

    @abstractmethod
    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        """
        Registers incoming tensor in tensor aggregator.

        :param x: Tensor to register.
        """

    def aggregate(self) -> Optional[NNCFTensor]:
        """
        Aggregates collected tensors and returns aggregated result.
        In case no tensors were collected returns None.

        :return: Aggregated result.
        """
        if self._collected_samples:
            return self._aggregate_impl()
        return None

    @abstractmethod
    def _aggregate_impl(self) -> NNCFTensor:
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


class TensorAggregatorBase(Aggregator, abc.ABC):
    """
    Tensor aggregator is designed to receive (register) calculated statistics and
    aggregate them in terms of NNCFTensor and NNCFTensorBackend opset.
    """

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples)

    def aggregate(self) -> Optional[NNCFTensor]:
        """
        Aggregates collected tensors and returns aggregated result.
        In case no tensors were collected returns None.

        :return: Aggregated result.
        """
        if self._collected_samples:
            return self._aggregate_impl()
        return None

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and self._num_samples == __o.num_samples

    def __hash__(self) -> int:
        return hash(self.__class__.__name__)


class OnlineTensorAggregator(TensorAggregatorBase, abc.ABC):
    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples)
        self._current_aggregate = None

    def _reset_sample_container(self):
        self._current_aggregate = None


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
        self._aggregators: Dict[AggregatorKey, TensorAggregatorBase] = {}
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
        aggregator: TensorAggregatorBase,
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

    def _aggregate(self) -> Dict[AggregatorKey, Optional[NNCFTensor]]:
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
        return self._stat_container(**kwargs)

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

    def replace_aggregator(self, key: Tuple[int, int, int], aggregator: TensorAggregatorBase) -> None:
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
        to a layout required for `register_input` method. This method is not a part of
        `register_input` to avoid all inputs passing to `TensorCollector.register_input` method.

        :param outputs: Target model outputs.
        :param output_info: Output info collected by a `TensorCollector.get_output_info` method.
        :returns: Model outputs in a format required by `TensorCollector.register_input` method.
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

    def __init__(self, tensor_collectors: List[TensorCollector]) -> None:
        """
        :param tensor_collectors: Tensor collectors to merge.
        """
        super().__init__()
        aggregators: Dict[Tuple[int, int], List[Tuple[TensorCollector, TensorAggregatorBase]]] = defaultdict(list)
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

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        return x


class MinReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        axis = self._get_axis(x)
        backend = x.backend
        return [backend.amin(x, axis=axis, keepdims=True)]


class MaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        axis = self._get_axis(x)
        backend = x.backend
        return [backend.amax(x, axis=axis, keepdims=True)]


class AbsMaxReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        backend = x.backend
        x = backend.abs(x)
        axis = self._get_axis(x)
        return [backend.amax(x, axis=axis, keepdims=True)]


class MeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        backend = x.backend
        axis = self._get_axis(x)
        return [backend.mean(x, axis, keepdims=True)]


class QuantileReducerBase(TensorReducerBase):
    def __init__(
        self,
        reduction_axes: Optional[ReductionAxes] = None,
        channel_axis: Optional[int] = None,
        quantile: Optional[Union[float, Tuple[float]]] = None,
        inplace: bool = False,
    ):
        super().__init__(reduction_axes=reduction_axes,
                         channel_axis=channel_axis,
                         inplace=False)
        self._quantile = (0.01, 0.99) if quantile is None else quantile

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.inplace, self._reduction_axes, tuple(self._quantile)))


class QuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        axis = self._get_axis(x)
        backend = x.backend
        return [t for t in backend.quantile(x, self._quantile, axis=axis, keepdims=True)]


class AbsQuantileReducer(QuantileReducerBase):
    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        backend = x.backend
        x = backend.abs(x)
        axis = self._get_axis(x)
        return [backend.quantile(x, self._quantile, axis=axis, keepdims=True)]


class BatchMeanReducer(TensorReducerBase):
    def __init__(self, inplace: bool = False):
        super().__init__(channel_axis=0, inplace=inplace)

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        backend = x.backend
        return [backend.mean(x, axis=0, keepdims=True)]


class MeanPerChReducer(TensorReducerBase):
    def __init__(self, channel_dim: int = 1, inplace: bool = False):
        super().__init__(channel_axis=channel_dim, inplace=inplace)

    def _reduce_out_of_place(self, x: List[NNCFTensor]) -> List[NNCFTensor]:
        x = x[0]
        backend = x.backend
        shape = x.shape
        axis = self._get_axis(x)
        retval = backend.mean(x, axis=axis, keepdims=True)
        return [retval]


##################################################Aggregators##################################################


class NoopAggregator(TensorAggregatorBase):
    # TODO (vshampor): this seems to be a bad design since this class does not actually aggregate anything
    #  and therefore cannot be an `Aggregator` instance. In particular this is visible in the interface violation
    #  for the _aggregate_impl function, which is supposed to return a single tensor - the aggregate - but has to
    #  return a list of tensors instead in order for the purpose of the class to be served.
    def _reset_sample_container(self):
        pass

    def __init__(self, num_samples: Optional[int]):
        super().__init__(num_samples)
        self._unaggregated_samples = []

    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        self._unaggregated_samples.append(x)

    def _aggregate_impl(self) -> List[NNCFTensor]:
        return self._unaggregated_samples


class ShapeAggregator(Aggregator):
    def __init__(self):
        super().__init__(num_samples=1)
        self._shape = None

    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        self._shape = x.shape

    def _aggregate_impl(self) -> List[int]:
        return self._shape

    def _reset_sample_container(self):
        self._shape = None


class MinAggregator(OnlineTensorAggregator):
    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        if self._current_aggregate is None:
            self._current_aggregate = x
        else:
            backend = x.backend
            self._current_aggregate = backend.minimum(x, self._current_aggregate)

    def _aggregate_impl(self) -> NNCFTensor:
        return self._current_aggregate


class MaxAggregator(OnlineTensorAggregator):
    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        if not self._current_aggregate:
            self._current_aggregate = x
        else:
            backend = x.backend
            self._current_aggregate = backend.maximum(x, self._current_aggregate)

    def _aggregate_impl(self) -> NNCFTensor:
        return self._current_aggregate


class OfflineAggregatorBase(TensorAggregatorBase, ABC):
    def __init__(self, use_per_sample_stats: bool = False, num_samples: Optional[int] = None, window_size: int = None):
        super().__init__(num_samples)
        self._window_size = window_size
        self._samples: Deque[NNCFTensor] = deque(maxlen=window_size)
        self._use_per_sample_stats = use_per_sample_stats

    def _register_reduced_input_impl(self, x: NNCFTensor) -> None:
        if self._use_per_sample_stats:
            backend = x.backend
            self._samples.extend(backend.unstack(x))
        else:
            self._samples.append(x)

    def _aggregate_impl(self) -> NNCFTensor:
        backend = next(iter(self._samples)).backend
        stacked_val = backend.stack(list(self._samples))
        return self._aggregate_stacked_samples(stacked_val)

    def _reset_sample_container(self):
        self._samples.clear()

    @abstractmethod
    def _aggregate_stacked_samples(self, stacked_samples: NNCFTensor) -> NNCFTensor:
        pass


class MeanAggregator(OfflineAggregatorBase):
    def _aggregate_stacked_samples(self, stacked_samples: NNCFTensor) -> NNCFTensor:
        return stacked_samples.mean(axis=0, keepdims=False)


class MedianAggregator(OfflineAggregatorBase):
    def _aggregate_stacked_samples(self, stacked_samples: NNCFTensor) -> NNCFTensor:
        return stacked_samples.median(axis=0, keepdims=False)


class NoOutliersAggregatorBase(OfflineAggregatorBase, ABC):
    def __init__(
        self,
        use_per_sample_stats: bool = False,
        num_samples: Optional[int] = None,
        window_size: int = None,
        quantile: float = 0.01,
    ):
        super().__init__(use_per_sample_stats, num_samples, window_size)
        self._quantile = quantile

    def _aggregate_stacked_samples(self, stacked_val: NNCFTensor) -> NNCFTensor:
        backend = stacked_val.backend
        alpha = self._quantile

        low_values, high_values = backend.quantile(stacked_val, [alpha, 1 - alpha], 0)
        outliers_mask = backend.logical_or(stacked_val < low_values, high_values < stacked_val)
        return self._aggregate_stacked_samples_with_no_outliers(stacked_val, outliers_mask)

    @abstractmethod
    def _aggregate_stacked_samples_with_no_outliers(
        self, stacked_val: NNCFTensor, outliers_mask: NNCFTensor
    ) -> NNCFTensor:
        pass

    def __eq__(self, __o: object) -> bool:
        return super().__eq__(__o) and self._quantile == __o._quantile

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self._quantile))


class MeanNoOutliersAggregator(NoOutliersAggregatorBase):
    def _aggregate_stacked_samples_with_no_outliers(
        self, stacked_val: NNCFTensor, outliers_mask: NNCFTensor
    ) -> NNCFTensor:
        backend = stacked_val.backend
        return backend.masked_mean(stacked_val, mask=outliers_mask, axis=0)


class MedianNoOutliersAggregator(NoOutliersAggregatorBase):
    def _aggregate_stacked_samples_with_no_outliers(
        self, stacked_val: NNCFTensor, outliers_mask: NNCFTensor
    ) -> NNCFTensor:
        backend = stacked_val.backend
        return backend.masked_median(stacked_val, mask=outliers_mask, axis=0)


AGGREGATORS_MAP = {
    AggregatorType.MIN: MinAggregator,
    AggregatorType.MAX: MaxAggregator,
    AggregatorType.MEAN: MeanAggregator,
    AggregatorType.MEAN_NO_OUTLIERS: MeanNoOutliersAggregator,
    AggregatorType.MEDIAN: MedianAggregator,
    AggregatorType.MEDIAN_NO_OUTLIERS: MedianNoOutliersAggregator,
}
