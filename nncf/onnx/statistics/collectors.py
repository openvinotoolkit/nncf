# Copyright (c) 2024 Intel Corporation
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
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import NoopReducer
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.quantization.advanced_parameters import StatisticsType


class ONNXBasicReducer(TensorReducerBase):
    def get_inplace_fn(self):
        raise NotImplementedError("ONNX backend has no support of inplace statistics yet.")


class ONNXMinReducer(ONNXBasicReducer, MinReducer):
    pass


class ONNXMaxReducer(ONNXBasicReducer, MaxReducer):
    pass


class ONNXAbsMaxReducer(ONNXBasicReducer, AbsMaxReducer):
    pass


class ONNXMeanReducer(ONNXBasicReducer, MeanReducer):
    pass


class ONNXQuantileReducer(ONNXBasicReducer, QuantileReducer):
    pass


class ONNXAbsQuantileReducer(ONNXBasicReducer, AbsQuantileReducer):
    pass


class ONNXBatchMeanReducer(ONNXBasicReducer, BatchMeanReducer):
    pass


class ONNXMeanPerChanelReducer(ONNXBasicReducer, MeanPerChReducer):
    pass


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
    noop_reducer = NoopReducer()

    kwargs = {
        "num_samples": num_samples,
        "window_size": window_size,
    }

    aggregate_mean = MeanAggregator(**kwargs)
    aggregate_shape = ShapeAggregator()

    collector = TensorCollector(MeanTensorStatistic)
    collector.register_statistic_branch(MeanTensorStatistic.MEAN_STAT, reducer, aggregate_mean)
    collector.register_statistic_branch(MeanTensorStatistic.SHAPE_STAT, noop_reducer, aggregate_shape)
    return collector


def get_raw_stat_collector(num_samples: int) -> TensorCollector:
    """
    Raw statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :return: Raw statistic collector.
    """
    reducer = RawReducer()
    aggregator = NoopAggregator(num_samples)

    collector = TensorCollector(RawTensorStatistic)
    collector.register_statistic_branch(RawTensorStatistic.VALUES_STATS, reducer, aggregator)
    return collector


ONNX_REDUCERS_MAP = {
    StatisticsType.MIN: ONNXMinReducer,
    StatisticsType.MAX: ONNXMaxReducer,
    StatisticsType.ABS_MAX: ONNXAbsMaxReducer,
    StatisticsType.MEAN: ONNXMeanReducer,
    StatisticsType.QUANTILE: ONNXQuantileReducer,
    StatisticsType.ABS_QUANTILE: ONNXAbsQuantileReducer,
}
