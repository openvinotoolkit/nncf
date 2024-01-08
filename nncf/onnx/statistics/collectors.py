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

from typing import Deque, List, Optional, Tuple, Union

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.onnx.statistics.statistics import ONNXMeanTensorStatistic
from nncf.onnx.statistics.statistics import ONNXRawTensorStatistic
from nncf.onnx.tensor import ONNXNNCFTensor
from nncf.quantization.advanced_parameters import StatisticsType


class ONNXNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for ONNXNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        return ONNXNNCFTensor(np.amin(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        return ONNXNNCFTensor(np.amax(x.tensor, axis=axis, keepdims=keepdims))

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
    def mean(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims=False) -> NNCFTensor:
        return ONNXNNCFTensor(np.median(x.tensor, axis=axis, keepdims=keepdims))

    @classmethod
    def masked_mean(
        cls,
        x: NNCFTensor,
        axis: Optional[Union[int, Tuple[int, ...], List[int]]],
        mask: Optional[NNCFTensor],
        keepdims: bool = False,
    ) -> NNCFTensor:
        if mask is None:
            return cls.mean(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return ONNXNNCFTensor(np.ma.mean(masked_x, axis=axis, keepdims=False).data)

    @classmethod
    def masked_median(
        cls,
        x: NNCFTensor,
        axis: Optional[Union[int, Tuple[int, ...], List[int]]],
        mask: Optional[NNCFTensor],
        keepdims: bool = False,
    ) -> NNCFTensor:
        if mask is None:
            return cls.median(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return ONNXNNCFTensor(np.ma.median(masked_x, axis=axis, keepdims=keepdims).data)

    @staticmethod
    def logical_or(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.logical_or(input_.tensor, other.tensor))

    @staticmethod
    def less(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(input_.tensor < other.tensor)

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return ONNXNNCFTensor(np.stack(x, axis=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        return [ONNXNNCFTensor(np.squeeze(e, axis)) for e in np.split(x.tensor, x.tensor.shape[axis], axis=axis)]

    @staticmethod
    def squeeze(x: NNCFTensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> NNCFTensor:
        return ONNXNNCFTensor(np.squeeze(x.tensor, axis=dim))

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return np.sum(tensor.tensor)

    @staticmethod
    def quantile(
        tensor: NNCFTensor,
        quantile: Union[float, List[float]],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        result = np.quantile(tensor.tensor, quantile, axis, keepdims=keepdims)
        return [ONNXNNCFTensor(x) for x in result]

    @classmethod
    def percentile(
        cls,
        tensor: NNCFTensor,
        percentile: Union[float, List[float]],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        raise NotImplementedError()

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return ONNXNNCFTensor(np.mean(x.tensor, axis=0))
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return ONNXNNCFTensor(np.mean(t, axis=(0, 2)))

    @staticmethod
    def transpose(x: NNCFTensor, axes: Tuple[int, ...]) -> NNCFTensor:
        return ONNXNNCFTensor(np.transpose(x.tensor, axes))

    @staticmethod
    def reshape(x: NNCFTensor, shape: Tuple[int, ...]) -> NNCFTensor:
        return ONNXNNCFTensor(np.reshape(x.tensor, shape))

    @staticmethod
    def cat(x: List[NNCFTensor], axis: int) -> NNCFTensor:
        x = [t.tensor for t in x]
        return ONNXNNCFTensor(np.concatenate(x, axis))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return ONNXNNCFTensor(np.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def sub(a: NNCFTensor, b: NNCFTensor) -> NNCFTensor:
        raise NotImplementedError()

    @staticmethod
    def zero_elements(x: NNCFTensor) -> NNCFTensor:
        raise NotImplementedError()


class ONNXNoopReducer(NoopReducer):
    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXMinReducer(MinReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXMaxReducer(MaxReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXAbsMaxReducer(AbsMaxReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXMeanReducer(MeanReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXQuantileReducer(QuantileReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXAbsQuantileReducer(AbsQuantileReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXBatchMeanReducer(BatchMeanReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


class ONNXMeanPerChanelReducer(MeanPerChReducer):
    def _get_processor(self):
        return ONNXNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return None  # TODO raise Error?

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return []


def get_mean_statistic_collector(
    num_samples: int, channel_axis: int, window_size: Optional[int] = None, inplace: bool = True
) -> TensorCollector:
    """
    Mean statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :param inplace: Whether the mean reducer should be calculated inplace or out of place.
    :return: Mean statistic collector.
    """
    inplace = False
    if channel_axis == 0:
        reducer = ONNXBatchMeanReducer(inplace)
    else:
        reducer = ONNXMeanPerChanelReducer(channel_axis=channel_axis, inplace=inplace)
    noop_reducer = ONNXNoopReducer()

    kwargs = {
        "tensor_processor": ONNXNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "window_size": window_size,
    }

    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(ONNXMeanTensorStatistic)
    collector.register_statistic_branch(ONNXMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(ONNXMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_raw_stat_collector(num_samples, inplace=False):
    reducer = ONNXNoopReducer()
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(ONNXRawTensorStatistic)
    collector.register_statistic_branch(ONNXRawTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector


ONNX_REDUCERS_MAP = {
    StatisticsType.MIN: ONNXMinReducer,
    StatisticsType.MAX: ONNXMaxReducer,
    StatisticsType.ABS_MAX: ONNXAbsMaxReducer,
    StatisticsType.MEAN: ONNXMeanReducer,
    StatisticsType.QUANTILE: ONNXQuantileReducer,
    StatisticsType.ABS_QUANTILE: ONNXAbsQuantileReducer,
}
