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

from typing import Any, Callable, Deque, List, Optional, Union

import numpy as np

from nncf.common.tensor import NNCFTensor
from nncf.common.tensor import TensorElementsType
from nncf.common.tensor_statistics.collectors import NNCFCollectorTensorProcessor
from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import InplaceInsertionFNType
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
from nncf.openvino.graph.node_utils import get_inplace_batch_mean_op
from nncf.openvino.graph.node_utils import get_inplace_max_op
from nncf.openvino.graph.node_utils import get_inplace_mean_op
from nncf.openvino.graph.node_utils import get_inplace_mean_per_ch
from nncf.openvino.graph.node_utils import get_inplace_min_op
from nncf.openvino.graph.node_utils import get_reducer_output_node_names
from nncf.openvino.graph.node_utils import get_result_node_name
from nncf.openvino.statistics.statistics import OVBatchTensorStatistic
from nncf.openvino.statistics.statistics import OVMeanTensorStatistic
from nncf.openvino.tensor import OVNNCFTensor
from nncf.quantization.advanced_parameters import StatisticsType


class OVNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for OVNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, tuple], keepdims: bool = True) -> NNCFTensor:
        return OVNNCFTensor(np.amin(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, tuple], keepdims: bool = True) -> NNCFTensor:
        return OVNNCFTensor(np.amax(x.tensor, axis=axis, keepdims=keepdims))

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
    def mean(x: NNCFTensor, axis: Union[int, tuple], keepdims: bool = False) -> NNCFTensor:
        return OVNNCFTensor(np.mean(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, tuple, list], keepdims: bool = False) -> NNCFTensor:
        return OVNNCFTensor(np.median(x.tensor, axis=axis, keepdims=keepdims))

    @classmethod
    def masked_mean(
        cls, x: NNCFTensor, axis: Optional[Union[int, tuple, list]], mask: Optional[NNCFTensor], keepdims: bool = False
    ) -> NNCFTensor:
        if mask is None:
            return cls.mean(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return OVNNCFTensor(np.ma.mean(masked_x, axis=axis, keepdims=False).data)

    @classmethod
    def masked_median(
        cls, x: NNCFTensor, axis: Optional[Union[int, tuple, list]], mask: Optional[NNCFTensor], keepdims: bool = False
    ) -> NNCFTensor:
        if mask is None:
            return cls.median(x, axis=axis, keepdims=keepdims)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        return OVNNCFTensor(np.ma.median(masked_x, axis=axis, keepdims=keepdims).data)

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return OVNNCFTensor(np.mean(x.tensor, axis=0))
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return OVNNCFTensor(np.mean(t, axis=(0, 2)))

    @classmethod
    def no_outliers_map(
        cls,
        x: NNCFTensor,
        fn: Callable[[NNCFTensor, int, NNCFTensor], Any],
        axis: int = 0,
        alpha: float = 0.01,
        keepdims: bool = False,
    ) -> NNCFTensor:
        if len(x.shape) == 1:
            return fn(x, axis=None, mask=None, keepdims=keepdims)

        x = x.tensor
        if axis:
            x = np.moveaxis(x, axis, 0)

        low_values, high_values = np.quantile(x, [alpha, 1 - alpha], 0)
        outliers_mask = np.logical_or(x < low_values, high_values < x)
        return fn(OVNNCFTensor(x), axis=0, mask=OVNNCFTensor(outliers_mask), keepdims=keepdims)

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

    @staticmethod
    def quantile(
        tensor: NNCFTensor, quantile: Union[float, List[float]], axis: Union[int, tuple, list], keepdims: bool = False
    ) -> List[NNCFTensor]:
        result = np.quantile(tensor.tensor, quantile, axis, keepdims=keepdims)
        return [OVNNCFTensor(x) for x in result]


class OVNoopReducer(NoopReducer):
    NAME = "noop"

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return [get_result_node_name(target_node_name, port_id)]


class OVMinReducer(MinReducer):
    NAME = "min"

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_min_op(self.NAME, self._reduction_shape)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVMaxReducer(MaxReducer):
    NAME = "max"

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_max_op(self.NAME, self._reduction_shape, False)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVAbsMaxReducer(AbsMaxReducer):
    NAME = "abs_max"

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_max_op(self.NAME, self._reduction_shape, True)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVMeanReducer(MeanReducer):
    NAME = "mean"

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_mean_op(self.NAME, self._reduction_shape)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVBatchMeanReducer(BatchMeanReducer):
    NAME = "batch_mean"

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_batch_mean_op(self.NAME)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVMeanPerChanelReducer(MeanPerChReducer):
    NAME = "mean_per_ch"

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_mean_per_ch(self.NAME, self._reduction_shape)

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVQuantileReducer(QuantileReducer):
    NAME = "quantile"

    @property
    def inplace(self):
        return False

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


class OVAbsQuantileReducer(AbsQuantileReducer):
    NAME = "abs_quantile"

    @property
    def inplace(self):
        return False

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def get_output_names(self, target_node_name: str, port_id: int) -> List[str]:
        return get_reducer_output_node_names(self.NAME, target_node_name, port_id, self.output_port_id, self.inplace)


def get_mean_stat_collector(num_samples, channel_axis, window_size=None, inplace=True):
    # TODO(dlyakhov): use inplace OVBatchMeanReducer and OVMeanPerChanelReducer
    # after migration on openvino-dev=2023.0
    inplace = False
    if channel_axis == 0:
        reducer = OVBatchMeanReducer(inplace)
    else:
        reducer = OVMeanPerChanelReducer(channel_axis, inplace)
    noop_reducer = OVNoopReducer()

    kwargs = {
        "tensor_processor": OVNNCFCollectorTensorProcessor,
        "use_per_sample_stats": False,
        "num_samples": num_samples,
        "window_size": window_size,
    }
    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(OVMeanTensorStatistic)
    collector.register_statistic_branch(OVMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(OVMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_mean_batch_stat_collector(num_samples, inplace=True):
    # TODO(dlyakhov): use inplace OVBatchMeanReducer
    # after migration on openvino-dev=2023.0
    inplace = False
    reducer = OVBatchMeanReducer(inplace=inplace)
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(OVBatchTensorStatistic)
    collector.register_statistic_branch(OVBatchTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector


OV_REDUCERS_MAP = {
    StatisticsType.MIN: OVMinReducer,
    StatisticsType.MAX: OVMaxReducer,
    StatisticsType.ABS_MAX: OVAbsMaxReducer,
    StatisticsType.MEAN: OVMeanReducer,
    StatisticsType.QUANTILE: OVQuantileReducer,
    StatisticsType.ABS_QUANTILE: OVAbsQuantileReducer,
}
