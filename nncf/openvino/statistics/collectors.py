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

from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple, Union

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
from nncf.experimental.common.tensor_statistics.collectors import OutputMetadata
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
from nncf.openvino.statistics.statistics import OVMeanTensorStatistic
from nncf.openvino.statistics.statistics import OVRawTensorStatistic
from nncf.openvino.tensor import OVNNCFTensor
from nncf.quantization.advanced_parameters import StatisticsType


class OVNNCFCollectorTensorProcessor(NNCFCollectorTensorProcessor):
    """
    A realization of the processing methods for OVNNCFTensors.
    """

    @staticmethod
    def reduce_min(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = True) -> NNCFTensor:
        return OVNNCFTensor(np.amin(x.tensor, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_max(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = True) -> NNCFTensor:
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
    def mean(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        comp_dtype, out_dtype = _get_computing_dtype(x.tensor.dtype)
        return OVNNCFTensor(
            np.mean(x.tensor, axis=axis, keepdims=keepdims, dtype=comp_dtype).astype(dtype=out_dtype, copy=False)
        )

    @staticmethod
    def median(x: NNCFTensor, axis: Union[int, Tuple[int, ...], List[int]], keepdims: bool = False) -> NNCFTensor:
        comp_dtype, out_dtype = _get_computing_dtype(x.tensor.dtype)
        t = x.tensor.astype(dtype=comp_dtype, copy=False)
        return OVNNCFTensor(np.median(t, axis=axis, keepdims=keepdims).astype(dtype=out_dtype, copy=False))

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
        comp_dtype, out_dtype = _get_computing_dtype(x.tensor.dtype)
        masked_x = np.ma.array(x.tensor, mask=mask.tensor)
        result = np.ma.mean(masked_x, axis=axis, keepdims=keepdims, dtype=comp_dtype)
        if isinstance(result, np.ma.MaskedArray):
            result = result.data
        return OVNNCFTensor(result.astype(dtype=out_dtype, copy=False))

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
        comp_dtype, out_dtype = _get_computing_dtype(x.tensor.dtype)
        t = x.tensor.astype(dtype=comp_dtype, copy=False)
        masked_x = np.ma.array(t, mask=mask.tensor)
        result = np.ma.median(masked_x, axis=axis, keepdims=keepdims)
        if isinstance(result, np.ma.MaskedArray):
            result = result.data
        return OVNNCFTensor(result.astype(dtype=out_dtype, copy=False))

    @staticmethod
    def mean_per_channel(x: NNCFTensor, axis: int) -> NNCFTensor:
        if len(x.shape) < 3:
            return OVNNCFCollectorTensorProcessor.mean(x, axis=0)
        x = np.moveaxis(x.tensor, axis, 1)
        t = x.reshape(x.shape[0], x.shape[1], -1)
        return OVNNCFCollectorTensorProcessor.mean(OVNNCFTensor(t), axis=(0, 2))

    @staticmethod
    def transpose(x: NNCFTensor, axes: Tuple[int, ...]) -> NNCFTensor:
        return OVNNCFTensor(np.transpose(x.tensor, axes))

    @staticmethod
    def reshape(x: NNCFTensor, shape: Tuple[int, ...]) -> NNCFTensor:
        return OVNNCFTensor(np.reshape(x.tensor, shape))

    @staticmethod
    def cat(x: List[NNCFTensor], axis: int) -> NNCFTensor:
        x = [t.tensor for t in x]
        return OVNNCFTensor(np.concatenate(x, axis))

    @staticmethod
    def batch_mean(x: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(np.mean(x.tensor, axis=0, keepdims=True))

    @staticmethod
    def logical_or(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(np.logical_or(input_.tensor, other.tensor))

    @staticmethod
    def less(input_: NNCFTensor, other: NNCFTensor) -> NNCFTensor:
        return OVNNCFTensor(input_.tensor < other.tensor)

    @staticmethod
    def stack(x: Union[List[NNCFTensor], Deque[NNCFTensor]], axis: int = 0) -> NNCFTensor:
        x = [t.tensor for t in x]
        return OVNNCFTensor(np.stack(x, axis=axis))

    @staticmethod
    def unstack(x: NNCFTensor, axis: int = 0) -> List[NNCFTensor]:
        return [OVNNCFTensor(np.squeeze(e, axis)) for e in np.split(x.tensor, x.tensor.shape[axis], axis=axis)]

    @staticmethod
    def squeeze(x: NNCFTensor, dim: Optional[Union[int, Tuple[int, ...]]] = None) -> NNCFTensor:
        return OVNNCFTensor(np.squeeze(x.tensor, axis=dim))

    @staticmethod
    def sum(tensor: NNCFTensor) -> TensorElementsType:
        return np.sum(tensor.tensor)

    @staticmethod
    def quantile(
        tensor: NNCFTensor,
        quantile: Union[float, List[float]],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[NNCFTensor]:
        result = np.quantile(tensor.tensor, quantile, axis, keepdims=keepdims)
        return [OVNNCFTensor(x) for x in result]

    @classmethod
    def percentile(
        cls,
        tensor: NNCFTensor,
        percentile: Union[float, List[float]],
        axis: Union[int, Tuple[int, ...], List[int]],
        keepdims: bool = False,
    ) -> List[TensorElementsType]:
        quantile = np.true_divide(percentile, 100)
        return cls.quantile(tensor, quantile=quantile, axis=axis, keepdims=keepdims)

    @staticmethod
    def sub(a: NNCFTensor, b: NNCFTensor) -> NNCFTensor:
        return NNCFTensor(a.tensor - b.tensor)

    @staticmethod
    def zero_elements(x: NNCFTensor) -> NNCFTensor:
        np_tensor = x.tensor
        eps = np.finfo(np_tensor.dtype).eps
        return NNCFTensor(np.abs(np_tensor) < eps)


@dataclass
class OVOutputMetadata(OutputMetadata):
    target_node_name: str
    port_id: int


class OVNoopReducer(NoopReducer):
    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return [get_result_node_name(output_metadata.target_node_name, output_metadata.port_id)]


class OVMinReducer(MinReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_min_op(self.name, self._reduction_axes)

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVMaxReducer(MaxReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_max_op(self.name, self._reduction_axes, False)

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVAbsMaxReducer(AbsMaxReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_max_op(self.name, self._reduction_axes, True)

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVMeanReducer(MeanReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_mean_op(self.name, self._reduction_axes)

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVBatchMeanReducer(BatchMeanReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_batch_mean_op(self.name)

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVMeanPerChanelReducer(MeanPerChReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self):
        return get_inplace_mean_per_ch(self.name, self._reduction_axes)

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVQuantileReducer(QuantileReducer):
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


class OVAbsQuantileReducer(AbsQuantileReducer):
    def _get_processor(self):
        return OVNNCFCollectorTensorProcessor

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None

    def get_output_names(self, output_metadata: OVOutputMetadata) -> List[str]:
        return get_reducer_output_node_names(
            self.name, output_metadata.target_node_name, output_metadata.port_id, self.output_port_id, self.inplace
        )


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
    # TODO(dlyakhov): use inplace OVBatchMeanReducer and OVMeanPerChanelReducer
    # after migration on openvino-dev=2023.0
    inplace = False
    if channel_axis == 0:
        reducer = OVBatchMeanReducer(inplace)
    else:
        reducer = OVMeanPerChanelReducer(channel_axis=channel_axis, inplace=inplace)
    noop_reducer = OVNoopReducer()

    kwargs = {
        "tensor_processor": OVNNCFCollectorTensorProcessor,
        "num_samples": num_samples,
        "window_size": window_size,
    }
    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(OVMeanTensorStatistic)
    collector.register_statistic_branch(OVMeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(OVMeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_raw_stat_collector(num_samples, inplace=False):
    reducer = OVNoopReducer()
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(OVRawTensorStatistic)
    collector.register_statistic_branch(OVRawTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector


OV_REDUCERS_MAP = {
    StatisticsType.MIN: OVMinReducer,
    StatisticsType.MAX: OVMaxReducer,
    StatisticsType.ABS_MAX: OVAbsMaxReducer,
    StatisticsType.MEAN: OVMeanReducer,
    StatisticsType.QUANTILE: OVQuantileReducer,
    StatisticsType.ABS_QUANTILE: OVAbsQuantileReducer,
}


def _get_computing_dtype(dtype: np.dtype) -> Tuple[Optional[np.dtype], Optional[np.dtype]]:
    """
    Determines the appropriate dtypes for intermediate computations and the final output,
    aiming to prevent overflow while maintaining precision.

    :param dtype: The dtype of the processed tensor.
    :return:
        - comp_dtype: The recommended dtype for intermediate computations to avoid overflow.
            If None, no dtype change is necessary for intermediate computations.
        - out_dtype: The recommended dtype for the final output, balancing precision and memory usage.
            If None, the input dtype is preserved for the output.
    """
    if dtype in [np.float32, np.float16]:
        return (np.float64, dtype)
    return (None, None)
