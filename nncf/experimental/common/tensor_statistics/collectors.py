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
from typing import Tuple, Optional, List, Set, Dict

from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import ReductionShape
from nncf.common.tensor import TensorType


class TensorReducerBase(ABC):

    def __init__(self,
                 reduction_shape: Optional[ReductionShape] = None,
                 inplace: bool = False):
        self._reduction_shape = reduction_shape
        self._tensor_processor: NNCFCollectorTensorProcessor = self._get_processor()
        self._inplace = inplace

    @property
    def inplace(self):
        return self._inplace

    @property
    def output_port_id(self) -> int:
        return 0

    @classmethod
    def name(cls):
        return cls.__name__

    @staticmethod
    @abstractmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
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

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and\
            self._reduction_shape == __o._reduction_shape and\
            self._inplace == __o.inplace

    def __hash__(self) -> int:
        return hash((self.name(), self._inplace))


class TensorAggregatorBase:
    def __init__(self, tensor_processor,
                 num_samples: Optional[int]):

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
        if self._num_samples is not None and \
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

    def __eq__(self, __o: object) -> bool:
        return isinstance(__o, self.__class__) and \
            self._num_samples == __o.num_samples

    def __hash__(self) -> int:
        return hash((self.name()))


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
        """
        Registers statistic collection branch for a container key. Correspondent input will be reduced
        by given reducer and reduced value will be registered and aggregated by given aggregator.
        Passed container key should be unique for the TensorCollector instance.
        Passed aggregator instance should never be used twice for one TensorCollector instance.

        :param container_key: Container key to pass aggregated statistic to.
        :param reducer: TensorReducer instance for the statistic collection branch.
        :param aggregator: TensorAggergator instance for the statistic collection branch.
        """
        if container_key in self._stat_container_kwargs_map:
            raise RuntimeError(f'Two differend statistic branches for one'
                               f' container key {container_key} are encountered')
        if any(aggr is aggregator for aggr in self._aggregators.values()):
            raise RuntimeError(f'One aggregator instance {aggregator} '
                               f' for different branches is encountered')

        self._reducers.add(reducer)
        key = (hash(reducer), hash(aggregator))

        if key not in self._aggregators:
            self._aggregators[key] = aggregator
        self._stat_container_kwargs_map[container_key] = key

    def get_output_info(self, target_node_name: str, port_id: int) -> List[str]:
        retval = []
        for reducer in self._reducers:
            retval.append((reducer.name(), reducer.get_output_name(target_node_name, port_id)))
        return retval

    def register_inputs(self, inputs: Dict[str, TensorType]):
        if not self._enabled:
            return

        reduced_inputs = {}
        for reducer in self._reducers:
            input_ = inputs[reducer.name()]
            reduced_input = reducer.reduce_input(input_)
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

    def get_inplace_fn_info(self):
        retval = []
        for reducer in self._reducers:
            if reducer.inplace:
                retval.append((reducer.get_inplace_fn(), reducer.output_port_id))
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


##################################################Reducers##################################################


class NoopReducer(TensorReducerBase):
    def __init__(self):
        super().__init__(inplace=False)

    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return None

    def get_inplace_fn(self):
        return []

    def _reduce_out_of_place(self, x: TensorType):
        return x


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

class BatchMeanReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType):
        return self._tensor_processor.batch_mean(x)


class MeanPerChReducer(TensorReducerBase):
    def _reduce_out_of_place(self, x: TensorType):
        return self._tensor_processor.mean_per_channel(x, self._reduction_shape)


class NumpyConverter(TensorReducerBase):
    def __init__(self):
        super().__init__(None, False)

    def _reduce_out_of_place(self, x: TensorType):
        return self._tensor_processor.to_numpy(x)


##################################################Aggregators##################################################


class NoopAggregator(TensorAggregatorBase):
    def __init__(self, num_samples: Optional[int]):
        super().__init__(None, num_samples)

    def _register_reduced_input_impl(self, x: TensorType):
        self._container.append(x.tensor)

    def aggregate(self):
        return self._container


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


class ShapeAggregator(TensorAggregatorBase):
    def __init__(self):
        super().__init__(None, 1)

    def _register_reduced_input_impl(self, x: TensorType):
        self._container = x

    def aggregate(self):
        return self._container.tensor.shape


class OfflineMinMaxAggregatorBase(TensorAggregatorBase):
    def __init__(self, tensor_processor, use_per_sample_stats: bool,
                 num_samples: Optional[int], window_size=None):
        super().__init__(tensor_processor, num_samples)
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
