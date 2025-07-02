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

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Optional

from nncf.common.utils.api_marker import api


@api()
class StatisticsType(Enum):
    """
    Enumeration of different types of statistics that are used to collect per sample
    statistics for activations and weights of the model.

    :param MAX: The maximum value in a tensor.
    :param MIN: The minimum value in a tensor.
    :param ABS_MAX: The maximum absolute value in a tensor.
    :param QUANTILE: A specific quantile value in a tensor.
    :param ABS_QUANTILE: A specific quantile value in the absolute tensor.
    :param MEAN: The mean value of a tensor.
    """

    MAX = "max"
    MIN = "min"
    ABS_MAX = "abs_max"
    QUANTILE = "quantile"
    ABS_QUANTILE = "abs_quantile"
    MEAN = "mean"


@api()
class AggregatorType(Enum):
    """
    Enumeration of different types of aggregators that are used to aggregate per sample
    statistics for activations and weights of the model.

    :param MEAN: The mean value of a set of tensors.
    :param MAX: The maximum value of a set of tensors.
    :param MIN: The minimum value of a set of tensors.
    :param MEDIAN: The median value of a set of tensors.
    :param MEAN_NO_OUTLIERS: The mean value of a set of tensors with outliers removed.
    :param MEDIAN_NO_OUTLIERS: The median value of a set of tensors with outliers removed.
    """

    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    MEDIAN = "median"
    MEAN_NO_OUTLIERS = "mean_no_outliers"
    MEDIAN_NO_OUTLIERS = "median_no_outliers"


@api()
@dataclass
class StatisticsCollectorParameters:
    """
    Contains parameters for collecting statistics for activations and weights of the model.

    :param statistics_type: The type of per sample statistics to collect.
    :type statistics_type: Optional[nncf.quantization.range_estimator.StatisticsType]
    :param aggregator_type: The type of aggregator of per sample statistics.
    :type aggregator_type: Optional[nncf.quantization.range_estimator.AggregatorType]
    :param clipping_value: The value to use for clipping the input tensors before
        collecting statistics.
    :type clipping_value: Optional[float]
    :param quantile_outlier_prob: The outlier probability for QUANTILE statistic or MEAN_NO_OUTLIERS/MEDIAN_NO_OUTLIERS
        aggregators. When using these together, please be aware that currently the quantile value will be applied to
        both.
    :type quantile_outlier_prob: float
    """

    statistics_type: Optional[StatisticsType] = None
    aggregator_type: Optional[AggregatorType] = None
    clipping_value: Optional[float] = None
    quantile_outlier_prob: float = 1e-4


@api()
@dataclass
class RangeEstimatorParameters:
    """
    Contains parameters for estimating the range of activations and weights of the model.

    :param min: The parameters for estimating the lower bound of the range.
    :type min: nncf.quantization.range_estimator.StatisticsCollectorParameters
    :param max: The Parameters for estimating the upper bound of the range.
    :type max: nncf.quantization.range_estimator.StatisticsCollectorParameters
    """

    min: StatisticsCollectorParameters = field(default_factory=StatisticsCollectorParameters)
    max: StatisticsCollectorParameters = field(default_factory=StatisticsCollectorParameters)


class RangeEstimatorParametersSet:
    """
    A class for specifying different sets of range estimator parameters.

    :param MINMAX: The range estimator parameters where the low bound of the range is
        calculated as global minimum of values of input tensors, the upper bound of
        the range as global maxima of the same values.
    :param MEAN_MINMAX: The range estimator parameters where the low bound of the range
        is calculated as average (across every sample) of minima of input tensors,
        the upper bound of the range as average of maxima of the same values.
    :param MEDIAN_MINMAX: The range estimator parameters where the low bound of the range
        is calculated as median (across every sample) of minima of input tensors,
        the upper bound of the range as median of maxima of the same values.
    :param MEAN_NO_OUTLIERS_MINMAX : The range estimator parameters where the low bound of the range
        is calculated as average (across all samples in range [min quantile, max quantile])
        of minima of input tensors, the upper bound of the range as average of maxima of the same values.
    :param MEAN_QUANTILE : The range estimator parameters where the low bound of the range
        is calculated as average (across every sample) of (quantile outlier probability)-quantiles,
        the upper bound of the range as average of (1 - quantile outlier probability)-quantiles of the same values.
    """

    MINMAX = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(statistics_type=StatisticsType.MIN, aggregator_type=AggregatorType.MIN),
        max=StatisticsCollectorParameters(statistics_type=StatisticsType.MAX, aggregator_type=AggregatorType.MAX),
    )

    MEAN_MINMAX = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(statistics_type=StatisticsType.MIN, aggregator_type=AggregatorType.MEAN),
        max=StatisticsCollectorParameters(statistics_type=StatisticsType.MAX, aggregator_type=AggregatorType.MEAN),
    )

    MEDIAN_MINMAX = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(statistics_type=StatisticsType.MIN, aggregator_type=AggregatorType.MEDIAN),
        max=StatisticsCollectorParameters(statistics_type=StatisticsType.MAX, aggregator_type=AggregatorType.MEDIAN),
    )

    MEAN_NO_OUTLIERS_MINMAX = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(
            statistics_type=StatisticsType.MIN, aggregator_type=AggregatorType.MEAN_NO_OUTLIERS
        ),
        max=StatisticsCollectorParameters(
            statistics_type=StatisticsType.MAX, aggregator_type=AggregatorType.MEAN_NO_OUTLIERS
        ),
    )

    MEAN_QUANTILE = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MEAN),
        max=StatisticsCollectorParameters(statistics_type=StatisticsType.QUANTILE, aggregator_type=AggregatorType.MEAN),
    )
