# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.openvino.graph.metatypes.groups import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS
from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.node_utils import get_number_if_op
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.openvino.quantization.backend_parameters import is_weight_compression_needed
from nncf.openvino.quantization.quantize_ifmodel import apply_algorithm_if_bodies
from nncf.openvino.rt_info import dump_parameters
from nncf.parameters import CompressWeightsMode
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import SensitivityMetric
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import convert_to_dict_recursively
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import calculate_accuracy_drop
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.quantize_model import quantize_with_tune_hyperparams
from nncf.quantization.quantize_model import warning_model_no_batchwise_support
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.scopes import IgnoredScope
from nncf.telemetry.decorator import tracked_function
from nncf.telemetry.events import NNCF_OV_CATEGORY

TTensor = TypeVar("TTensor")


@tracked_function(NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset"])
def native_quantize_if_op_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> ov.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend via the OpenVINO Runtime API.
    """
    if not fast_bias_correction:
        raise NotImplementedError(
            "The BiasCorrection algorithm is not supported for OpenVINO models with If operation."
        )
    quantization_algorithm = PostTrainingQuantization(
        mode=mode,
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )

    graph = GraphConverter.create_nncf_graph(model)
    warning_model_no_batchwise_support(graph, advanced_parameters, model_type, OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS)
    if_ops_number = get_number_if_op(model)
    all_models_number = if_ops_number * 2 + 1
    nncf_logger.info(
        f"The model consists of {if_ops_number} If node(-s) with then and else bodies. \
            Main model and all If bodies will be quantized recursively."
    )
    quantized_model, _ = apply_algorithm_if_bodies(
        quantization_algorithm, model, graph, calibration_dataset, subset_size, 1, all_models_number
    )

    if is_weight_compression_needed(advanced_parameters):
        compress_quantize_weights_transformation(quantized_model)

    dump_parameters(
        quantized_model,
        {
            "preset": preset,
            "target_device": target_device.value,
            "subset_size": subset_size,
            "fast_bias_correction": fast_bias_correction,
            "model_type": model_type,
            "ignored_scope": ignored_scope,
            "advanced_parameters": convert_to_dict_recursively(advanced_parameters),
        },
    )
    return quantized_model


@tracked_function(NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset"])
def native_quantize_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> ov.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend via the OpenVINO Runtime API.
    """
    quantization_algorithm = PostTrainingQuantization(
        mode=mode,
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )
    graph = GraphConverter.create_nncf_graph(model)
    warning_model_no_batchwise_support(graph, advanced_parameters, model_type, OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS)
    quantized_model = quantization_algorithm.apply(model, graph, dataset=calibration_dataset)

    if is_weight_compression_needed(advanced_parameters):
        compress_quantize_weights_transformation(quantized_model)

    dump_parameters(
        quantized_model,
        {
            "preset": preset,
            "target_device": target_device.value,
            "subset_size": subset_size,
            "fast_bias_correction": fast_bias_correction,
            "model_type": model_type,
            "ignored_scope": ignored_scope,
            "advanced_parameters": convert_to_dict_recursively(advanced_parameters),
        },
    )
    return quantized_model


@tracked_function(
    NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset", "max_drop", "drop_type"]
)
def native_quantize_with_accuracy_control_impl(
    model: ov.Model,
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
) -> ov.Model:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend via the
    OpenVINO Runtime API.
    """
    if advanced_accuracy_restorer_parameters is None:
        advanced_accuracy_restorer_parameters = AdvancedAccuracyRestorerParameters()

    compress_weights = is_weight_compression_needed(advanced_quantization_parameters)

    if advanced_quantization_parameters is None:
        copied_parameters = AdvancedQuantizationParameters()
    else:
        copied_parameters = deepcopy(advanced_quantization_parameters)
    copied_parameters.backend_params[BackendParameters.COMPRESS_WEIGHTS] = False

    quantized_model = quantize_impl(
        model=model,
        calibration_dataset=calibration_dataset,
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=copied_parameters,
    )

    if advanced_accuracy_restorer_parameters.intermediate_model_dir:
        quantized_model_path = f"{advanced_accuracy_restorer_parameters.intermediate_model_dir}/intermediate_model.xml"
        ov.serialize(quantized_model, quantized_model_path)

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

    # TODO(andrey-churkin): Collect statistics only once
    if advanced_accuracy_restorer_parameters.tune_hyperparams and not should_terminate:
        model = remove_friendly_name_duplicates(model)
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
            copied_parameters,
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

    if compress_weights:
        compress_quantize_weights_transformation(quantized_model)

    dump_parameters(
        quantized_model,
        {
            "preset": preset,
            "target_device": target_device.value,
            "subset_size": subset_size,
            "fast_bias_correction": fast_bias_correction,
            "model_type": model_type,
            "ignored_scope": ignored_scope,
            "max_drop": max_drop,
            "drop_type": drop_type.value,
            "advanced_quantization_parameters": convert_to_dict_recursively(advanced_quantization_parameters),
            "advanced_accuracy_restorer_parameters": convert_to_dict_recursively(advanced_accuracy_restorer_parameters),
        },
    )
    return quantized_model


def quantize_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> ov.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend.
    """
    model = remove_friendly_name_duplicates(model)

    quantize_fn = native_quantize_impl
    if get_number_if_op(model) > 0:
        quantize_fn = native_quantize_if_op_impl

    return quantize_fn(
        model=model,
        calibration_dataset=calibration_dataset,
        mode=mode,
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )


def wrap_validation_fn(validation_fn):
    """
    Wraps validation function to support case when it only returns metric value.

    :param validation_fn: Validation function to wrap.
    :return: Wrapped validation function.
    """

    def wrapper(*args, **kwargs):
        retval = validation_fn(*args, **kwargs)
        if isinstance(retval, tuple):
            return retval
        return retval, None

    return wrapper


def quantize_with_accuracy_control_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], float],
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
) -> ov.Model:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend.
    """

    quantize_with_accuracy_control_fn = native_quantize_with_accuracy_control_impl

    val_func = wrap_validation_fn(validation_fn)

    return quantize_with_accuracy_control_fn(
        model,
        calibration_dataset,
        validation_dataset,
        val_func,
        max_drop,
        drop_type,
        preset,
        target_device,
        subset_size,
        fast_bias_correction,
        model_type,
        ignored_scope,
        advanced_quantization_parameters,
        advanced_accuracy_restorer_parameters,
    )


def compress_weights_impl(
    model: ov.Model,
    dataset: Dataset,
    mode: CompressWeightsMode,
    ratio: float,
    group_size: int,
    ignored_scope: IgnoredScope,
    all_layers: bool,
    sensitivity_metric: SensitivityMetric,
    awq: bool,
    subset_size: int,
) -> ov.Model:
    """
    Implementation of the `compress_weights()` method for the OpenVINO backend.
    """

    model = remove_friendly_name_duplicates(model)
    compression_algorithm = WeightCompression(
        mode, ratio, group_size, ignored_scope, all_layers, sensitivity_metric, awq, subset_size
    )
    graph = NNCFGraphFactory.create(model)
    return compression_algorithm.apply(model, graph, dataset=dataset)
