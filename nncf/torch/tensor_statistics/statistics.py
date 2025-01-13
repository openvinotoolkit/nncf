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


from nncf.experimental.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.experimental.common.tensor_statistics.statistics import TensorStatistic


def pt_convert_stat_to_min_max_tensor_stat(statistic: TensorStatistic) -> MinMaxTensorStatistic:
    if isinstance(statistic, MinMaxTensorStatistic):
        return statistic
    if isinstance(statistic, MedianMADTensorStatistic):
        # Using three-sigma approach to estimate min and max
        # Constant factor depends on the distribution form - assuming normal and the factor is 1.4826
        return MinMaxTensorStatistic(
            min_values=statistic.median_values - 3 * 1.4826230 * statistic.mad_values,
            max_values=statistic.median_values + 3 * 1.4826230 * statistic.mad_values,
        )
    if isinstance(statistic, PercentileTensorStatistic):
        if len(statistic.percentile_vs_values_dict.keys()) < 2:
            raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
        min_pct = min(statistic.percentile_vs_values_dict.keys())
        max_pct = max(statistic.percentile_vs_values_dict.keys())
        return MinMaxTensorStatistic(
            min_values=statistic.percentile_vs_values_dict[min_pct],
            max_values=statistic.percentile_vs_values_dict[max_pct],
        )
    raise ValueError("Unknown TensorStatistic to generate min-max stat from!")
