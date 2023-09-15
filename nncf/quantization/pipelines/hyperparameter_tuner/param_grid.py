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

import itertools
from typing import Any, Dict

from nncf.common.quantization.structs import QuantizationPreset
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import StatisticsCollectorParameters
from nncf.quantization.range_estimator import StatisticsType


def get_quantization_param_grid() -> Dict[str, Any]:
    """
    Returns params grid for post-training quantization algorithm.
    """
    min_param_values = [
        StatisticsCollectorParameters(
            statistics_type=StatisticsType.MIN,
            aggregator_type=AggregatorType.MIN,
        ),
        StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE,
            aggregator_type=AggregatorType.MEAN,
            quantile_outlier_prob=10e-4,
        ),
        StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE,
            aggregator_type=AggregatorType.MEAN,
            quantile_outlier_prob=10e-5,
        ),
    ]
    max_param_values = [
        StatisticsCollectorParameters(
            statistics_type=StatisticsType.MAX,
            aggregator_type=AggregatorType.MAX,
        ),
        StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE,
            aggregator_type=AggregatorType.MEAN,
            quantile_outlier_prob=10e-4,
        ),
        StatisticsCollectorParameters(
            statistics_type=StatisticsType.QUANTILE,
            aggregator_type=AggregatorType.MEAN,
            quantile_outlier_prob=10e-5,
        ),
    ]

    param_grid = {
        "preset": [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
        "fast_bias_correction": [True, False],
        "advanced_parameters:weights_range_estimator_params": [
            RangeEstimatorParameters(
                min=StatisticsCollectorParameters(statistics_type=StatisticsType.MIN),
                max=StatisticsCollectorParameters(statistics_type=StatisticsType.MAX),
            )
        ],
        "advanced_parameters:activations_range_estimator_params": [
            RangeEstimatorParameters(min=min_v, max=max_v)
            for min_v, max_v in itertools.product(min_param_values, max_param_values)
        ],
    }

    return param_grid
