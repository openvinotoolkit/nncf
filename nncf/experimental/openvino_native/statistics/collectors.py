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

from typing import Union, List, Deque, Tuple, Callable
from abc import abstractmethod
import openvino.runtime.opset9 as opset

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import TensorReducerBase
from nncf.common.tensor_statistics.collectors import MinMaxStatisticCollector
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
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
from nncf.common.tensor_statistics.collectors import BatchStatisticCollector
from nncf.experimental.openvino_native.tensor import OVNNCFTensor
from nncf.experimental.openvino_native.statistics.statistics import OVMinMaxTensorStatistic
from nncf.experimental.openvino_native.statistics.statistics import OVMeanTensorStatistic
from nncf.experimental.openvino_native.statistics.statistics import OVBatchTensorStatistic
from nncf.experimental.openvino_native.graph.node_utils import get_reduce_node_name
from nncf.experimental.openvino_native.graph.node_utils import get_result_node_name
from nncf.experimental.openvino_native.graph.node_utils import get_inplace_reduce_op
from nncf.common.tensor import TensorType


class OVNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for OVNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return OVNNCFTensor(np.amin(x.tensor, axis=axis, keepdims=True))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return OVNNCFTensor(np.amax(x.tensor, axis=axis, keepdims=True))

    @staticmethod
    def abs(x: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(np.abs(x.tensor))

    @staticmethod
    def min(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(np.minimum(x1.tensor, x2.tensor))

    @staticmethod
    def max(x1: NNCFTensor, x2: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(np.maximum(x1.tensor, x2.tensor))

    @staticmethod
    def mean(x: NNCFTensor, axis: Union[int, tuple]) -> NNCFTensor:
        return OVNNCFTensor(np.mean(x.tensor, axis=axis))

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return OVNNCFTensor(np.mean(x.tensor, axis=0))
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return OVNNCFTensor(np.mean(t, axis=(0, 2)))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(np.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return OVNNCFTensor(np.stack(x, axis=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        return [OVNNCFTensor(np.squeeze(e, axis)) for e in np.split(x.tensor, x.tensor.shape[axis], axis=axis)]

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return np.sum(tensor.tensor)


class OVMinMaxStatisticCollector(MinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    def get_statistics(self) -> OVMinMaxTensorStatistic:
        return OVMinMaxTensorStatistic(self._min_values.tensor, self._max_values.tensor)


class OVMeanMinMaxStatisticCollector(MeanMinMaxStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    def get_statistics(self) -> OVMinMaxTensorStatistic:
        return OVMinMaxTensorStatistic(self._min_aggregate().tensor, self._max_aggregate().tensor)


class OVMeanStatisticCollector(MeanStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    def get_statistics(self) -> OVMeanTensorStatistic:
        return OVMeanTensorStatistic(self._mean_aggregate().tensor, self._shape())


class OVBatchStatisticCollector(BatchStatisticCollector):
    @staticmethod
    def _get_processor() -> NNCFCollectorTensorProcessor:
        return OVNNCFCollectorTensorProcessor()

    def get_statistics(self) -> OVBatchTensorStatistic:
        return OVBatchTensorStatistic(self._all_values)


class OVNoopReducer(NoopReducer):
    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        return get_result_node_name(target_node_name, port_id)


class OVMinReducer(MinReducer):
    NAME = 'min'
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_reduce_op(opset.reduce_min, self.NAME,
                                     self._reduction_shape, False)

    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        if self.inplace:
            target_node_name = get_reduce_node_name(target_node_name, self.NAME)
        return get_result_node_name(target_node_name, port_id)


class OVMaxReducer(MaxReducer):
    NAME = 'max'
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_reduce_op(opset.reduce_max, self.NAME,
                                     self._reduction_shape, False)

    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        if self.inplace:
            target_node_name = get_reduce_node_name(target_node_name, self.NAME)
        return get_result_node_name(target_node_name, port_id)


class OVAbsMaxReducer(AbsMaxReducer):
    NAME = 'max'
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_reduce_op(opset.reduce_max, self.NAME,
                                     self._reduction_shape, True)

    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        if self.inplace:
            target_node_name = get_reduce_node_name(target_node_name, self.NAME)
        return get_result_node_name(target_node_name, port_id)


class OVBatchMeanReducer(BatchMeanReducer):
    NAME = 'batch_mean'
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_reduce_op(opset.reduce_mean, self.NAME,
                                     0, False)

    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        if self.inplace:
            target_node_name = get_reduce_node_name(target_node_name, self.NAME)
        return get_result_node_name(target_node_name, port_id)


class OVMeanPerChanelReducer(MeanPerChReducer):
    NAME = 'mean_per_ch'
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        raise NotImplementedError()

    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        #if self.inplace:
        #    target_node_name = get_reduce_node_name(target_node_name, self.NAME)
        return get_result_node_name(target_node_name, port_id)


def get_min_max_stat_collector(num_samples, reduction_shape, use_abs_max, inplace):
    reduce_min = OVMinReducer(reduction_shape, inplace)
    if use_abs_max:
        reduce_max = OVAbsMaxReducer(reduction_shape, inplace)
    else:
        reduce_max = OVMaxReducer(reduction_shape, inplace)

    kwargs = {
        'num_samples': num_samples,
        'tensor_processor': OVNNCFCollectorTensorProcessor
    }
    aggregate_min = OnlineMinAggregator(**kwargs)
    aggregate_max = OnlineMaxAggregator(**kwargs)

    collector = TensorCollector(OVMinMaxTensorStatistic)
    collector.add_branch(OVMinMaxTensorStatistic.MIN_STAT, reduce_min, aggregate_min)
    collector.add_branch(OVMinMaxTensorStatistic.MAX_STAT, reduce_max, aggregate_max)
    return collector


def get_mean_min_max_stat_collector(num_samples, reduction_shape, use_abs_max,
                                    use_per_sample_stats, inplace, window_size=None):
    reduce_min = OVMinReducer(reduction_shape, inplace)
    if use_abs_max:
        reduce_max = OVAbsMaxReducer(reduction_shape, inplace)
    else:
        reduce_max = OVMaxReducer(reduction_shape, inplace)

    kwargs = {
        'tensor_processor': OVNNCFCollectorTensorProcessor,
        'use_per_sample_stats': use_per_sample_stats,
        'num_samples': num_samples,
        'window_size': window_size
    }
    aggregate_min = OfflineMeanAggregator(**kwargs)
    aggregate_max = OfflineMeanAggregator(**kwargs)

    collector = TensorCollector(OVMinMaxTensorStatistic)
    collector.add_branch(OVMinMaxTensorStatistic.MIN_STAT, reduce_min, aggregate_min)
    collector.add_branch(OVMinMaxTensorStatistic.MAX_STAT, reduce_max, aggregate_max)
    return collector


def get_mean_stat_collector(num_samples, reduction_shape, window_size=None, inplace=False):
    reducer_cls = OVBatchMeanReducer if reduction_shape == 0 else OVMeanPerChanelReducer
    reducer = reducer_cls(reduction_shape, inplace)
    noop_reducer = OVNoopReducer()

    kwargs = {
        'tensor_processor': OVNNCFCollectorTensorProcessor,
        'use_per_sample_stats': False,
        'num_samples': num_samples,
        'window_size': window_size
    }
    aggregate_mean = OfflineMeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(OVMeanTensorStatistic)
    collector.add_branch(OVMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.add_branch(OVMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_mean_batch_stat_collector(num_samples, inplace=True):
    reducer = OVBatchMeanReducer(inplace=inplace)
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(OVBatchTensorStatistic)
    collector.add_branch(OVBatchTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector
