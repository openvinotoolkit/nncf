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

import torch

from nncf.common.tensor_statistics.statistics import TensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic


class PTMinMaxTensorStatistic(MinMaxTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-6) -> bool:
        return bool(torch.allclose(tensor1, tensor2, rtol=rtol))


class PTMedianMADTensorStatistic(MedianMADTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-6) -> bool:
        return bool(torch.allclose(tensor1, tensor2, rtol=rtol))


class PTPercentileTensorStatistic(PercentileTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol=1e-6) -> bool:
        return bool(torch.allclose(tensor1, tensor2, rtol=rtol))


def pt_convert_stat_to_min_max_tensor_stat(statistic: TensorStatistic) -> PTMinMaxTensorStatistic:
    if isinstance(statistic, PTMinMaxTensorStatistic):
        return statistic
    if isinstance(statistic, PTMedianMADTensorStatistic):
        # Using three-sigma approach to estimate min and max
        # Constant factor depends on the distribution form - assuming normal and the factor is 1.4826
        return PTMinMaxTensorStatistic(statistic.median_values - 3 * 1.4826230 * statistic.mad_values,
                                     statistic.median_values + 3 * 1.4826230 * statistic.mad_values)
    if isinstance(statistic, PTPercentileTensorStatistic):
        if len(statistic.percentile_vs_values_dict.keys()) < 2:
            raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
        min_pct = min(statistic.percentile_vs_values_dict.keys())
        max_pct = max(statistic.percentile_vs_values_dict.keys())
        return PTMinMaxTensorStatistic(statistic.percentile_vs_values_dict[min_pct],
                                     statistic.percentile_vs_values_dict[max_pct])
    raise ValueError("Unknown TensorStatistic to generate min-max stat from!")
