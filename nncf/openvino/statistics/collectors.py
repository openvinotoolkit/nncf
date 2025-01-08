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

from typing import Optional

from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AbsQuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import InplaceInsertionFNType
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanVarianceReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import ShapeReducer
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.openvino.graph.node_utils import get_inplace_batch_mean_op
from nncf.openvino.graph.node_utils import get_inplace_max_op
from nncf.openvino.graph.node_utils import get_inplace_max_var_op
from nncf.openvino.graph.node_utils import get_inplace_mean_max_op
from nncf.openvino.graph.node_utils import get_inplace_mean_op
from nncf.openvino.graph.node_utils import get_inplace_mean_per_ch
from nncf.openvino.graph.node_utils import get_inplace_mean_var_op
from nncf.openvino.graph.node_utils import get_inplace_min_op
from nncf.openvino.graph.node_utils import get_inplace_shape_op
from nncf.quantization.advanced_parameters import StatisticsType


class OVMinReducer(MinReducer):

    def get_inplace_fn(self):
        return get_inplace_min_op(self._reduction_axes)


class OVMaxReducer(MaxReducer):

    def get_inplace_fn(self):
        return get_inplace_max_op(self._reduction_axes, False)


class OVAbsMaxReducer(AbsMaxReducer):

    def get_inplace_fn(self):
        return get_inplace_max_op(self._reduction_axes, True)


class OVMeanReducer(MeanReducer):

    def get_inplace_fn(self):
        return get_inplace_mean_op(self._reduction_axes)


class OVMeanVarianceReducer(MeanVarianceReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_var_op(self._reduction_axes)


class OVMaxVarianceReducer(MaxVarianceReducer):
    def get_inplace_fn(self):
        return get_inplace_max_var_op(self._reduction_axes)


class OVMeanAbsMaxReducer(MeanAbsMaxReducer):
    def get_inplace_fn(self):
        return get_inplace_mean_max_op(self._reduction_axes, True)


class OVShapeReducer(ShapeReducer):
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return get_inplace_shape_op()


class OVBatchMeanReducer(BatchMeanReducer):

    def get_inplace_fn(self):
        return get_inplace_batch_mean_op()


class OVMeanPerChanelReducer(MeanPerChReducer):

    def get_inplace_fn(self):
        return get_inplace_mean_per_ch(self._channel_axis)


class OVQuantileReducer(QuantileReducer):
    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None


class OVAbsQuantileReducer(AbsQuantileReducer):

    def get_inplace_fn(self) -> Optional[InplaceInsertionFNType]:
        return None


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
    if channel_axis == 0:
        reducer = OVBatchMeanReducer(inplace)
    else:
        reducer = OVMeanPerChanelReducer(channel_axis=channel_axis, inplace=inplace)
    raw_reducer = RawReducer()

    kwargs = {
        "num_samples": num_samples,
        "window_size": window_size,
    }
    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(MeanTensorStatistic)
    collector.register_statistic_branch(MeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(MeanTensorStatistic.SHAPE_STAT, raw_reducer, aggregate_shape)
    return collector


def get_raw_stat_collector(num_samples: Optional[int] = None) -> TensorCollector:
    reducer = RawReducer()
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(RawTensorStatistic)
    collector.register_statistic_branch(RawTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector


OV_REDUCERS_MAP = {
    StatisticsType.MIN: OVMinReducer,
    StatisticsType.MAX: OVMaxReducer,
    StatisticsType.ABS_MAX: OVAbsMaxReducer,
    StatisticsType.MEAN: OVMeanReducer,
    StatisticsType.QUANTILE: OVQuantileReducer,
    StatisticsType.ABS_QUANTILE: OVAbsQuantileReducer,
}
