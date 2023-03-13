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

from typing import Union, List, Deque

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import BatchStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxOfflineStatisticCollectorSpec
from nncf.common.tensor_statistics.collectors import TensorCollector
from nncf.common.tensor_statistics.collectors import NoopReducer
from nncf.common.tensor_statistics.collectors import MinReducer
from nncf.common.tensor_statistics.collectors import MaxReducer
from nncf.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.common.tensor_statistics.collectors import OnlineMinAggregator
from nncf.common.tensor_statistics.collectors import OnlineMaxAggregator
from nncf.common.tensor_statistics.collectors import OfflineMinAggregator
from nncf.common.tensor_statistics.collectors import OfflineMaxAggregator
from nncf.common.tensor_statistics.collectors import OfflineMeanAggregator
from nncf.common.tensor_statistics.collectors import ShapeAggregator
from nncf.common.tensor_statistics.collectors import NoopAggregator
from nncf.common.tensor_statistics.collectors import TensorAggregatorBase
from nncf.common.tensor_statistics.collectors import MeanStatisticCollector
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.common.tensor_statistics.collectors import MeanMinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import MeanStatisticCollector
from nncf.onnx.tensor import ONNXNNCFTensor
from nncf.onnx.graph.node_utils import get_inplace_min_op
from nncf.onnx.graph.node_utils import get_inplace_max_op
from nncf.onnx.graph.node_utils import get_inplace_mean_op
from nncf.onnx.graph.node_utils import get_output_edge_name
from nncf.onnx.statistics.statistics import ONNXMinMaxTensorStatistic
from nncf.onnx.statistics.statistics import ONNXMeanTensorStatistic
from nncf.onnx.statistics.statistics import ONNXBatchTensorStatistic


class ONNXNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for ONNXNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return ONNXNNCFTensor(np.amin(x.tensor, axis=axis))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return ONNXNNCFTensor(np.amax(x.tensor, axis=axis))

    @staticmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.abs(x.tensor))

    @staticmethod
    def min(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.minimum(x1.tensor, x2.tensor))

    @staticmethod
    def max(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.maximum(x1.tensor, x2.tensor))

    @staticmethod
    def mean(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=axis))

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return ONNXNNCFTensor(np.mean(x.tensor, axis=0))
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return ONNXNNCFTensor(np.mean(t, axis=(0, 2)))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return ONNXNNCFTensor(np.stack(x, axis=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        return [ONNXNNCFTensor(np.squeeze(e, axis)) for e in np.split(x.tensor, x.tensor.shape[axis], axis=axis)]

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return np.sum(tensor.tensor)


class ONNXMinMaxStatisticCollector(MinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMinMaxTensorStatistic:
        return ONNXMinMaxTensorStatistic(self._min_values.tensor, self._max_values.tensor)


class ONNXMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMinMaxTensorStatistic:
        return ONNXMinMaxTensorStatistic(self._min_aggregate().tensor, self._max_aggregate().tensor)


class ONNXMeanStatisticCollector(MeanStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXMeanTensorStatistic:
        return ONNXMeanTensorStatistic(self._mean_aggregate().tensor, self._shape())


class ONNXBatchStatisticCollector(BatchStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return ONNXNNCFCollectorTensorProcessor()

    def _register_input(self, x: ONNXNNCFTensor):
        self._register_input_common(x)

    def _get_statistics(self) -> ONNXBatchTensorStatistic:
        return ONNXBatchTensorStatistic(self._all_values)


class ONNXNoopReducer(NoopReducer):
    def get_output_name(self, target_edge_name: str, port_id: int) -> str:
        return target_edge_name


class ONNXMinReducer(MinReducer):
    NAME = 'MinReduce'
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_min_op(self._reduction_shape)

    def get_output_name(self, target_edge_name: str, port_id: int) -> str:
        if self.inplace:
            return get_output_edge_name(target_edge_name, self.NAME)
        return target_edge_name


class ONNXMaxReducer(MaxReducer):
    NAME = 'MaxReduce'
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_max_op(self._reduction_shape, False)

    def get_output_name(self, target_edge_name: str, port_id: int) -> str:
        if self.inplace:
            return get_output_edge_name(target_edge_name, self.NAME)
        return target_edge_name


class ONNXAbsMaxReducer(AbsMaxReducer):
    NAME = 'MaxReduce'
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_max_op(self._reduction_shape, True)

    def get_output_name(self, target_edge_name: str, port_id: int) -> str:
        if self.inplace:
            return get_output_edge_name(target_edge_name, self.NAME)
        return target_edge_name


class ONNXBatchMeanReducer(BatchMeanReducer):
    NAME = 'MeanReduce'
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_mean_op(0)

    def get_output_name(self, target_edge_name: str, port_id: int) -> str:
        if self.inplace:
            return get_output_edge_name(target_edge_name, self.NAME)
        return target_edge_name


class ONNXMeanPerChanelReducer(MeanPerChReducer):
    NAME = 'mean_per_ch'
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        raise NotImplementedError()

    def get_output_name(self, target_edge_name: str, port_id: int) -> str:
        #if self.inplace:
        #    target_node_name = get_reduce_node_name(target_node_name, self.NAME)
        return target_edge_name


def get_min_max_stat_collector(num_samples, reduction_shape, use_abs_max, inplace):
    reduce_min = ONNXMinReducer(reduction_shape, inplace)
    if use_abs_max:
        reduce_max = ONNXAbsMaxReducer(reduction_shape, inplace)
    else:
        reduce_max = ONNXMaxReducer(reduction_shape, inplace)

    kwargs = {
        'num_samples': num_samples,
        'tensor_processor': ONNXNNCFCollectorTensorProcessor
    }
    aggregate_min = OnlineMinAggregator(**kwargs)
    aggregate_max = OnlineMaxAggregator(**kwargs)

    collector = TensorCollector(ONNXMinMaxTensorStatistic)
    collector.add_branch(ONNXMinMaxTensorStatistic.MIN_STAT, reduce_min, aggregate_min)
    collector.add_branch(ONNXMinMaxTensorStatistic.MAX_STAT, reduce_max, aggregate_max)
    return collector


def get_mean_min_max_stat_collector(num_samples, reduction_shape, use_abs_max,
                                    use_per_sample_stats, inplace, window_size=None):
    reduce_min = ONNXMinReducer(reduction_shape, inplace)
    if use_abs_max:
        reduce_max = ONNXAbsMaxReducer(reduction_shape, inplace)
    else:
        reduce_max = ONNXMaxReducer(reduction_shape, inplace)

    kwargs = {
        'tensor_processor': ONNXNNCFCollectorTensorProcessor,
        'use_per_sample_stats': use_per_sample_stats,
        'num_samples': num_samples,
        'window_size': window_size
    }
    aggregate_min = OfflineMeanAggregator(**kwargs)
    aggregate_max = OfflineMeanAggregator(**kwargs)

    collector = TensorCollector(ONNXMinMaxTensorStatistic)
    collector.add_branch(ONNXMinMaxTensorStatistic.MIN_STAT, reduce_min, aggregate_min)
    collector.add_branch(ONNXMinMaxTensorStatistic.MAX_STAT, reduce_max, aggregate_max)
    return collector


def get_mean_stat_collector(num_samples, reduction_shape, window_size=None, inplace=False):
    reducer_cls = ONNXBatchMeanReducer if reduction_shape == 0 else ONNXMeanPerChanelReducer
    reducer = reducer_cls(reduction_shape, inplace)
    noop_reducer = ONNXNoopReducer()

    kwargs = {
        'tensor_processor': ONNXNNCFCollectorTensorProcessor,
        'use_per_sample_stats': False,
        'num_samples': num_samples,
        'window_size': window_size
    }
    aggregate_mean = OfflineMeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(ONNXMeanTensorStatistic)
    collector.add_branch(ONNXMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.add_branch(ONNXMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_mean_batch_stat_collector(num_samples, inplace=True):
    reducer = ONNXBatchMeanReducer(inplace=inplace)
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(ONNXBatchMeanReducer)
    collector.add_branch(ONNXBatchMeanReducer.VALUES_STATS, reducer, aggregator)
    return collector
