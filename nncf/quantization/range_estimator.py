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

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Optional


class StatisticsType(Enum):
    MAX = 'max'
    MIN = 'min'
    ABS_MAX = 'abs_max'
    QUANTILE = 'quantile'
    ABS_QUANTILE = 'abs_quantile'
    MEAN = 'mean'


class AggregatorType(Enum):
    MEAN = 'mean'
    MAX = 'max'
    MIN = 'min'
    MEDIAN = 'median'
    MEAN_NO_OUTLIERS = 'mean_no_outliers'
    MEDIAN_NO_OUTLIERS = 'median_no_outliers'


@dataclass
class StatisticsCollectorParameters:
    statistics_type: Optional[StatisticsType] = None
    aggregator_type: Optional[AggregatorType] = None
    clipping_value: Optional[float] = None
    quantile_outlier_prob: float = 1e-4


@dataclass
class RangeEstimatorParameters:
    min: StatisticsCollectorParameters = field(
        default_factory=StatisticsCollectorParameters)
    max: StatisticsCollectorParameters = field(
        default_factory=StatisticsCollectorParameters)


class RangeEstimatorParametersSet:
    MINMAX = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(
            statistics_type=StatisticsType.MIN,
            aggregator_type=AggregatorType.MIN
        ),
        max=StatisticsCollectorParameters(
            statistics_type=StatisticsType.MAX,
            aggregator_type=AggregatorType.MAX
        )
    )

    MEAN_MINMAX = RangeEstimatorParameters(
        min=StatisticsCollectorParameters(
            statistics_type=StatisticsType.MIN,
            aggregator_type=AggregatorType.MEAN
        ),
        max=StatisticsCollectorParameters(
            statistics_type=StatisticsType.MAX,
            aggregator_type=AggregatorType.MEAN
        )
    )
