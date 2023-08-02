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
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

from nncf.common.quantization.structs import QuantizationPreset
from nncf.data.dataset import Dataset
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults
from nncf.quantization.algorithms.hyperparameter_tuner.algorithm import HyperparameterTuner
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.range_estimator import AggregatorType
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import StatisticsCollectorParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.scopes import IgnoredScope

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


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


def quantize_with_tune_hyperparams(
    model: TModel,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
    initial_metric_results: MetricResults,
    quantized_metric_results: MetricResults,
    tuner_subset_size: int = 300,
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> TModel:
    """
    Applies hyperparameters tuning for post-training quantization algorithm.
    """
    init_quantization_params = {
        "preset": preset,
        "target_device": target_device,
        "subset_size": subset_size,
        "fast_bias_correction": fast_bias_correction,
        "model_type": model_type,
        "ignored_scope": ignored_scope,
        "advanced_parameters": advanced_quantization_parameters,
    }

    quantization_param_grid = get_quantization_param_grid()

    hyperparameter_tuner = HyperparameterTuner(
        PostTrainingQuantization,
        init_quantization_params,
        quantization_param_grid,
        calibration_dataset,
        validation_fn,
        tuner_subset_size,
        initial_metric_results,
        quantized_metric_results,
    )

    quantized_model = hyperparameter_tuner.apply(model, validation_dataset)

    return quantized_model
