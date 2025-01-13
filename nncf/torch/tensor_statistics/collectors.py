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

from functools import partial
from typing import Optional, Tuple, Type

import numpy as np

from nncf.experimental.common.tensor_statistics.collectors import AbsMaxReducer
from nncf.experimental.common.tensor_statistics.collectors import AggregatorBase
from nncf.experimental.common.tensor_statistics.collectors import BatchMeanReducer
from nncf.experimental.common.tensor_statistics.collectors import MaxAggregator
from nncf.experimental.common.tensor_statistics.collectors import MaxReducer
from nncf.experimental.common.tensor_statistics.collectors import MeanAggregator
from nncf.experimental.common.tensor_statistics.collectors import MeanPerChReducer
from nncf.experimental.common.tensor_statistics.collectors import MedianAbsoluteDeviationAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinAggregator
from nncf.experimental.common.tensor_statistics.collectors import MinReducer
from nncf.experimental.common.tensor_statistics.collectors import NoopAggregator
from nncf.experimental.common.tensor_statistics.collectors import PercentileAggregator
from nncf.experimental.common.tensor_statistics.collectors import QuantileReducer
from nncf.experimental.common.tensor_statistics.collectors import RawReducer
from nncf.experimental.common.tensor_statistics.collectors import ShapeAggregator
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MeanTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import RawTensorStatistic
from nncf.tensor import Tensor


def _reshape_all(targets: Tuple[Tensor, ...], target_shape: Tuple[int, ...]):
    return map(lambda stat: stat.reshape(target_shape), targets)


def _get_wrapped_min_max_tensor_statistic(target_shape: Tuple[int, ...]) -> Type[MinMaxTensorStatistic]:
    """
    Returns MinMaxTensorStatistic type but all statistics are reshaped to target_shape.

    :param target_shape: Target shape of the tensor statistic
    :return: MinMaxTensorStatistic type but all statistics are reshaped to target_shape.
    """

    class WrappedPTMinMaxTensorStatistic(MinMaxTensorStatistic):
        def __init__(self, min_values, max_values):
            min_values, max_values = _reshape_all((min_values, max_values), target_shape)
            super().__init__(min_values, max_values)

    return WrappedPTMinMaxTensorStatistic


def _get_wrapped_percentile_tensor_statistic(target_shape: Tuple[int, ...]) -> Type[PercentileTensorStatistic]:
    """
    Returns PercentileTensorStatistic type but all statistics are reshaped to target_shape.

    :param target_shape: Target shape of the tensor statistic
    :return: PercentileTensorStatistic type but all statistics are reshaped to target_shape.
    """

    class WrappedPTPercentileTensorStatistic(PercentileTensorStatistic):
        def __init__(self, percentile_vs_values_dict):
            reshaped_percentiles = {}
            for k, v in percentile_vs_values_dict.items():
                reshaped_percentiles[k] = v.reshape(target_shape)
            super().__init__(reshaped_percentiles)

    return WrappedPTPercentileTensorStatistic


def get_min_max_statistic_collector(
    use_abs_max: bool,
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
) -> TensorCollector:
    """
    Min max statistic collector builder.

    :param use_abs_max: Whether to use abs max reducer or max reducer.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :return: Min max statistic collector.
    """

    tensor_collector = TensorCollector(_get_wrapped_min_max_tensor_statistic(target_shape=scale_shape))

    aggregator_kwargs = {
        "num_samples": num_samples,
        "aggregation_axes": aggregation_axes,
    }
    min_reducer = MinReducer(reduction_axes)
    min_aggregator = MinAggregator(**aggregator_kwargs)
    tensor_collector.register_statistic_branch(MinMaxTensorStatistic.MIN_STAT, min_reducer, min_aggregator)

    max_reducer_cls = AbsMaxReducer if use_abs_max else MaxReducer
    max_reducer = max_reducer_cls(reduction_axes)
    max_aggregator = MaxAggregator(**aggregator_kwargs)
    tensor_collector.register_statistic_branch(MinMaxTensorStatistic.MAX_STAT, max_reducer, max_aggregator)
    return tensor_collector


def get_mixed_min_max_statistic_collector(
    use_abs_max: bool,
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    use_means_of_mins: bool,
    use_means_of_maxs: bool,
    num_samples: int = None,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Mixed min max statistic collector builder.

    :param use_abs_max: Whether to use abs max reducer or max reducer.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param use_means_of_mins: Whether to use mean or min aggregator for minimum statistic branch.
    :param use_means_of_maxs: Whether to use mean or max aggregator for maximum statistic branch.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Mixed min max statistic collector.
    """
    tensor_collector = TensorCollector(_get_wrapped_min_max_tensor_statistic(target_shape=scale_shape))
    min_reducer = MinReducer(reduction_axes)

    kwargs = {
        "num_samples": num_samples,
        "aggregation_axes": aggregation_axes,
        "window_size": window_size,
    }
    min_aggregator_cls = MeanAggregator if use_means_of_mins else MinAggregator
    min_aggregator = min_aggregator_cls(**kwargs)
    tensor_collector.register_statistic_branch(MinMaxTensorStatistic.MIN_STAT, min_reducer, min_aggregator)

    max_reducer_cls = AbsMaxReducer if use_abs_max else MaxReducer
    max_reducer = max_reducer_cls(reduction_axes)
    max_aggregator_cls = MeanAggregator if use_means_of_maxs else MaxAggregator
    max_aggregator = max_aggregator_cls(**kwargs)
    tensor_collector.register_statistic_branch(MinMaxTensorStatistic.MAX_STAT, max_reducer, max_aggregator)

    return tensor_collector


def get_median_mad_statistic_collector(
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Median Absolute Deviation statistic collector builder.

    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Median Absolute Deviation statistic collector.

    """

    class WrappedPTMedianMADTensorStatistic(MedianMADTensorStatistic):
        def __init__(self, median_values, mad_values):
            median_values, mad_values = _reshape_all((median_values, mad_values), scale_shape)
            super().__init__(median_values, mad_values)

    return _get_collection_without_reduction(
        MedianAbsoluteDeviationAggregator,
        WrappedPTMedianMADTensorStatistic,
        reduction_axes=reduction_axes,
        aggregation_axes=aggregation_axes,
        num_samples=num_samples,
        window_size=window_size,
    )


def get_percentile_tensor_collector(
    percentiles_to_collect: Tuple[int, ...],
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Percentile statistic collector builder.

    :param percentiles_to_collect: Percentiles to use on aggregation phase.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Percentile statistic collector.
    """
    return _get_collection_without_reduction(
        partial(PercentileAggregator, percentiles_to_collect=percentiles_to_collect),
        _get_wrapped_percentile_tensor_statistic(target_shape=scale_shape),
        reduction_axes=reduction_axes,
        aggregation_axes=aggregation_axes,
        num_samples=num_samples,
        window_size=window_size,
    )


def _get_collection_without_reduction(
    aggregator_cls: AggregatorBase,
    statistic_cls: AggregatorBase,
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Helper function to build a tensor collector which is reducing statistics exclusively during aggregation phase.

    :param aggregator_cls: Aggregator class to build the tensor collector.
    :param aggregator_cls: Statistic class to build the tensor collector.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Target statistic collector.
    """
    tensor_collector = TensorCollector(statistic_cls)
    reducer = RawReducer()
    aggregation_axes = list(set(list(aggregation_axes) + [dim + 1 for dim in reduction_axes]))
    aggregator = aggregator_cls(
        aggregation_axes=aggregation_axes,
        window_size=window_size,
        num_samples=num_samples,
    )

    tensor_collector.register_statistic_branch(
        MedianMADTensorStatistic.TENSOR_STATISTIC_OUTPUT_KEY, reducer, aggregator
    )
    return tensor_collector


def get_mean_percentile_statistic_collector(
    percentiles_to_collect: Tuple[int, ...],
    reduction_axes: Tuple[int, ...],
    aggregation_axes: Tuple[int, ...],
    scale_shape: Tuple[int, ...],
    num_samples: int,
    window_size: Optional[int] = None,
) -> TensorCollector:
    """
    Mean percentile statistic collector builder.

    :param percentiles_to_collect: Percentiles to use on reduction phase.
    :param reduction_axes: Axes to use in reduction functions.
    :param aggregation_axes: Axes to use in aggregation functions.
    :param scale_shape: Target shape for collected statistics.
    :param num_samples: Maximum number of samples to collect.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Mean percentile statistic collector.
    """
    tensor_collector = TensorCollector(_get_wrapped_percentile_tensor_statistic(target_shape=scale_shape))
    quantiles_to_collect = np.true_divide(percentiles_to_collect, 100)
    reducer = QuantileReducer(reduction_axes=reduction_axes, quantile=quantiles_to_collect)
    for output_port_id, p in enumerate(percentiles_to_collect):
        aggregator = MeanAggregator(
            aggregation_axes=aggregation_axes,
            num_samples=num_samples,
            window_size=window_size,
        )
        tensor_collector.register_statistic_branch(
            (PercentileTensorStatistic.PERCENTILE_VS_VALUE_DICT, p), reducer, aggregator, output_port_id
        )
    return tensor_collector


def get_mean_statistic_collector(
    num_samples: int, channel_axis: int, window_size: Optional[int] = None
) -> TensorCollector:
    """
    Mean statistic collector builder.

    :param num_samples: Maximum number of samples to collect.
    :param channel_axis: Channel axis to use during reduction phase.
    :param window_size: Number of samples from the end of the list of collected samples to aggregate.
        Aggregates all available collected statistics in case parameter is None.
    :return: Mean statistic collector.
    """
    if channel_axis == 0:
        reducer = BatchMeanReducer()
    else:
        reducer = MeanPerChReducer(channel_axis=channel_axis)
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
