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

import tensorflow as tf

from nncf.common.tensor_statistics.statistics import MedianMADTensorStatistic
from nncf.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.common.tensor_statistics.statistics import PercentileTensorStatistic
from nncf.common.tensor_statistics.statistics import TensorStatistic


class TFMinMaxTensorStatistic(MinMaxTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: tf.Tensor, tensor2: tf.Tensor, rtol=1e-6) -> bool:
        return bool(tf.experimental.numpy.allclose(tensor1, tensor2, rtol=rtol))


class TFMedianMADTensorStatistic(MedianMADTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: tf.Tensor, tensor2: tf.Tensor, rtol=1e-6) -> bool:
        return bool(tf.experimental.numpy.allclose(tensor1, tensor2, rtol=rtol))


class TFPercentileTensorStatistic(PercentileTensorStatistic):
    @staticmethod
    def tensor_eq(tensor1: tf.Tensor, tensor2: tf.Tensor, rtol=1e-6) -> bool:
        return bool(tf.experimental.numpy.allclose(tensor1, tensor2, rtol=rtol))


def tf_convert_stat_to_min_max_tensor_stat(statistic: TensorStatistic) -> TFMinMaxTensorStatistic:
    if isinstance(statistic, TFMinMaxTensorStatistic):
        return statistic
    if isinstance(statistic, TFMedianMADTensorStatistic):
        # Using three-sigma approach to estimate min and max
        # Constant factor depends on the distribution form - assuming normal and the factor is 1.4826
        return TFMinMaxTensorStatistic(
            statistic.median_values - 3 * 1.4826230 * statistic.mad_values,
            statistic.median_values + 3 * 1.4826230 * statistic.mad_values,
        )
    if isinstance(statistic, TFPercentileTensorStatistic):
        if len(statistic.percentile_vs_values_dict.keys()) < 2:
            raise ValueError("Cannot create a min-max statistic for less than 2 percentile values")
        min_pct = min(statistic.percentile_vs_values_dict.keys())
        max_pct = max(statistic.percentile_vs_values_dict.keys())
        return TFMinMaxTensorStatistic(
            statistic.percentile_vs_values_dict[min_pct], statistic.percentile_vs_values_dict[max_pct]
        )
    raise ValueError("Unknown TensorStatistic to generate min-max stat from!")
