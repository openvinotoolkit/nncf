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

from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import onnx

import nncf
from nncf.common.logging.logger import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.onnx.graph.metatypes.groups import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import calculate_accuracy_drop
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.quantize_model import quantize_with_tune_hyperparams
from nncf.quantization.quantize_model import warning_model_no_batchwise_support
from nncf.scopes import IgnoredScope

TTensor = TypeVar("TTensor")


def quantize_impl(
    model: onnx.ModelProto,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> onnx.ModelProto:
    """
    Implementation of the `quantize()` method for the ONNX backend.
    """
    if target_device == TargetDevice.CPU_SPR:
        raise nncf.ValidationError("target_device == CPU_SPR is not supported.")
    if mode is not None:
        raise ValueError(f"mode={mode} is not supported")
    if model.opset_import[0].version < 10:
        raise nncf.ValidationError("ONNX models with opset version < 10 do not support quantization.")
    if model.opset_import[0].version < 13:
        nncf_logger.warning(
            "ONNX models with 10 < opset version < 13 do not support per-channel quantization."
            " Per-tensor quantization will be applied."
        )
        if advanced_parameters is None:
            advanced_parameters = AdvancedQuantizationParameters()
        advanced_parameters.weights_quantization_params = QuantizationParameters(per_channel=False)
        advanced_parameters.activations_quantization_params = QuantizationParameters(per_channel=False)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        ignored_scope=ignored_scope,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        advanced_parameters=advanced_parameters,
    )

    graph = GraphConverter.create_nncf_graph(model)
    warning_model_no_batchwise_support(graph, advanced_parameters, model_type, OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS)
    quantized_model = quantization_algorithm.apply(model, graph, dataset=calibration_dataset)

    return quantized_model


def quantize_with_accuracy_control_impl(
    model: onnx.ModelProto,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], Tuple[float, Union[None, List[float], List[List[TTensor]]]]],
    max_drop: float = 0.01,
    drop_type: DropType = DropType.ABSOLUTE,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters] = None,
    advanced_accuracy_restorer_parameters: Optional[AdvancedAccuracyRestorerParameters] = None,
) -> onnx.ModelProto:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the ONNX backend.
    """
    if advanced_accuracy_restorer_parameters is None:
        advanced_accuracy_restorer_parameters = AdvancedAccuracyRestorerParameters()

    if advanced_quantization_parameters is None:
        advanced_quantization_parameters = AdvancedQuantizationParameters()

    quantized_model = quantize_impl(
        model=model,
        calibration_dataset=calibration_dataset,
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_quantization_parameters,
    )

    if advanced_accuracy_restorer_parameters.intermediate_model_dir:
        quantized_model_path = f"{advanced_accuracy_restorer_parameters.intermediate_model_dir}/intermediate_model.onnx"
        onnx.save(quantized_model, quantized_model_path)

    evaluator = Evaluator(validation_fn)
    evaluator.enable_iteration_count()

    initial_metric_results = evaluator.collect_metric_results(model, validation_dataset, model_name="initial")

    validation_dataset_size = evaluator.num_passed_iterations
    evaluator.disable_iteration_count()

    quantized_metric_results = evaluator.collect_metric_results(
        quantized_model, validation_dataset, model_name="quantized"
    )

    should_terminate, accuracy_drop = calculate_accuracy_drop(
        initial_metric_results.metric_value, quantized_metric_results.metric_value, max_drop, drop_type
    )

    nncf_logger.info(f"Accuracy drop: {accuracy_drop} ({drop_type})")

    if advanced_accuracy_restorer_parameters.tune_hyperparams and not should_terminate:
        tuned_quantized_model = quantize_with_tune_hyperparams(
            model,
            calibration_dataset,
            validation_dataset,
            validation_fn,
            initial_metric_results,
            quantized_metric_results,
            subset_size,
            preset,
            target_device,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_quantization_parameters,
        )
        tuned_quantized_metric_results = evaluator.collect_metric_results(
            tuned_quantized_model, validation_dataset, model_name="tuned"
        )
        should_terminate, tuned_accuracy_drop = calculate_accuracy_drop(
            initial_metric_results.metric_value, tuned_quantized_metric_results.metric_value, max_drop, drop_type
        )

        nncf_logger.info(f"Accuracy drop (tuned): {tuned_accuracy_drop} ({drop_type})")

        if should_terminate or tuned_accuracy_drop < accuracy_drop:
            quantized_model = tuned_quantized_model
            quantized_metric_results = tuned_quantized_metric_results

    if not should_terminate:
        ranking_subset_size = subset_size
        if advanced_accuracy_restorer_parameters.ranking_subset_size is not None:
            ranking_subset_size = advanced_accuracy_restorer_parameters.ranking_subset_size

        accuracy_restorer = QuantizationAccuracyRestorer(
            ranking_subset_size,
            advanced_accuracy_restorer_parameters.max_num_iterations,
            max_drop,
            drop_type,
            advanced_accuracy_restorer_parameters.num_ranking_workers,
            advanced_accuracy_restorer_parameters.restore_mode,
        )
        quantized_model = accuracy_restorer.apply(
            model,
            initial_metric_results,
            quantized_model,
            quantized_metric_results,
            validation_dataset,
            validation_dataset_size,
            evaluator,
        )

    return quantized_model
