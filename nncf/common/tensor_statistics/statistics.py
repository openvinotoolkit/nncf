"""
 Copyright (c) 2021 Intel Corporation
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

from abc import ABC


class TensorStatistic(ABC):
    pass


class MinMaxTensorStatistic(TensorStatistic):
    def __init__(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values


class MedianMADTensorStatistic(TensorStatistic):
    def __init__(self, median_values, mad_values):
        self.median_values = median_values
        self.mad_values = mad_values


class PercentileTensorStatistic(TensorStatistic):
    def __init__(self, percentile_vs_values_dict):
        self.percentile_vs_values_dict = percentile_vs_values_dict


def convert_stat_to_min_max_tensor_stat(statistic: TensorStatistic) -> TensorStatistic:
    if isinstance(statistic, MinMaxTensorStatistic):
        return statistic
    if isinstance(statistic, MedianMADTensorStatistic):
        # Using three-sigma approach to estimate min and max
        # Constant factor depends on the distribution form - assuming normal and the factor is 1.4826
        return MinMaxTensorStatistic(statistic.median_values - 3 * 1.4826230 * statistic.mad_values,
                                     statistic.median_values + 3 * 1.4826230 * statistic.mad_values)
    if isinstance(statistic, PercentileTensorStatistic):
        if len(statistic.percentile_vs_values_dict.keys()) < 2:
            raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
        min_pct = min(statistic.percentile_vs_values_dict.keys())
        max_pct = max(statistic.percentile_vs_values_dict.keys())
        return MinMaxTensorStatistic(statistic.percentile_vs_values_dict[min_pct],
                                     statistic.percentile_vs_values_dict[max_pct])
    raise ValueError("Unknown TensorStatistic to generate min-max stat from!")
