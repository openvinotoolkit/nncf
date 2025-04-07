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

import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import onnx

import nncf
from nncf.common.logging.logger import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.onnx.graph.metatypes.groups import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS
from nncf.onnx.graph.model_utils import eliminate_nop_cast
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


def quantize_pre_process(model: onnx.ModelProto, save_as_external_data: bool = True):
    """
    Preprocesses the provided ONNX model for quantization.

    This method performs the following steps:
        1. Infers shapes in the model.
        2. Removes redundant 'No-op' cast nodes from the model.

    :param model: The ONNX model to be preprocessed.
    :param save_as_external_data: A boolean flag indicating whether to
        save the model with external data. If `True`, external data is
        saved separately; otherwise, the model is saved as a single file.
    :return: A preprocessed ONNX model, ready for quantization.
    """
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as temp_dir:
        temp_path = Path(temp_dir)
        input_model_path = str(temp_path / "input_model.onnx")

        if save_as_external_data:
            onnx.save_model(
                model,
                input_model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location="model.data",
                size_threshold=1024,
                convert_attribute=False,
            )
        else:
            onnx.save(model, input_model_path)
        model = None

        shape_inferred_model_path = str(temp_path / "shape_inferred_model.onnx")
        onnx.shape_inference.infer_shapes_path(input_model_path, shape_inferred_model_path)

        preprocessed_model = onnx.load(shape_inferred_model_path)
        preprocessed_model = eliminate_nop_cast(preprocessed_model)

    return preprocessed_model


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
        msg = "target_device == CPU_SPR is not supported."
        raise nncf.ValidationError(msg)
    if mode is not None:
        msg = f"mode={mode} is not supported"
        raise ValueError(msg)
    if model.opset_import[0].version < 10:
        msg = "ONNX models with opset version < 10 do not support quantization."
        raise nncf.ValidationError(msg)
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

    model = quantize_pre_process(model)
    graph = GraphConverter.create_nncf_graph(model, infer_shapes=False)
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
