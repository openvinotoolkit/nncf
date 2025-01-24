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

import itertools
from typing import Any, Dict, List

from nncf.common.quantization.quantizer_propagation.structs import QuantizerPropagationRule
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.backend import BackendType
from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.channel_alignment.algorithm import ChannelAlignment
from nncf.quantization.algorithms.fast_bias_correction.algorithm import FastBiasCorrection
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization
from nncf.quantization.algorithms.pipeline import Pipeline
from nncf.quantization.algorithms.smooth_quant.algorithm import SmoothQuant
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import StatisticsCollectorParameters
from nncf.quantization.range_estimator import StatisticsType

ParamGrid = Dict[str, List[Any]]


def _get_minmax_quantization_param_grid() -> ParamGrid:
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
        "advanced_parameters:quantizer_propagation_rule": [
            QuantizerPropagationRule.MERGE_IF_ALL_BRANCHES_SAME,
        ],
        "preset": [QuantizationPreset.PERFORMANCE, QuantizationPreset.MIXED],
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


def _get_smooth_quant_param_grid() -> ParamGrid:
    alpha_values = [0.15, 0.25, 0.5, 0.75, 0.95]
    return {
        "advanced_parameters:smooth_quant_alphas": [
            AdvancedSmoothQuantParameters(matmul=alpha_v) for alpha_v in itertools.product(alpha_values)
        ]
    }


def _get_channel_alignment_param_grid() -> ParamGrid:
    return {}


def _get_bias_correction_param_grid() -> ParamGrid:
    return {"fast_bias_correction": [True, False]}


def get_quantization_param_grids(pipeline: Pipeline, backend: BackendType) -> List[ParamGrid]:
    """
    Returns params grid for post-training quantization algorithm.
    """
    algorithm_cls_to_param_grid = {
        SmoothQuant: _get_smooth_quant_param_grid(),
        ChannelAlignment: _get_channel_alignment_param_grid(),
        MinMaxQuantization: _get_minmax_quantization_param_grid(),
        FastBiasCorrection: _get_bias_correction_param_grid(),
        BiasCorrection: _get_bias_correction_param_grid(),
    }

    param_grids = []
    for step in pipeline.pipeline_steps:
        param_grid = {}
        for algorithm in step:
            if backend not in algorithm.available_backends:
                continue
            param_grid.update(algorithm_cls_to_param_grid[algorithm.__class__])
        if param_grid:
            param_grids.append(param_grid)

    return param_grids
