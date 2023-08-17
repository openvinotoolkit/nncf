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

import importlib
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.openvino.quantization.backend_parameters import is_weight_compression_needed
from nncf.openvino.quantization.weights_compression import insert_pre_compression_operations
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import convert_to_dict_recursively
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import calculate_accuracy_drop
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.quantize_model import quantize_with_tune_hyperparams
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.scopes import IgnoredScope
from nncf.telemetry.decorator import tracked_function
from nncf.telemetry.events import NNCF_OV_CATEGORY

USE_POT_AS_DEFAULT = False

TTensor = TypeVar("TTensor")


def should_use_pot(advanced_parameters: Optional[AdvancedQuantizationParameters]) -> bool:
    """
    Returns True if POT should be used for quantization, False otherwise.

    :param advanced_parameters: Advanced quantization parameters.
    :return: True if POT should be used, False otherwise.
    :raises ImportError if POT is not found in the Python environment.
    """
    use_pot = USE_POT_AS_DEFAULT
    if advanced_parameters is not None:
        use_pot = advanced_parameters.backend_params.get(BackendParameters.USE_POT, USE_POT_AS_DEFAULT)

    if not use_pot:
        return False

    try:
        importlib.import_module("openvino.tools.pot")
    except ImportError:
        nncf_logger.error(
            "OpenVINO POT was not found in your Python environment.\n"
            "Please install the openvino-dev package, e.g. via pypi: pip install openvino-dev.\n"
        )

    return True


def dump_parameters(model: ov.Model, parameters: Dict, path: Optional[List] = None) -> None:
    """
    Dumps input parameters into Model's meta section.

    :param model: ov.Model instance.
    :param parameters: Incoming dictionary with parameters to save.
    :param path: Optional list of the paths.
    """
    try:
        path = path if path else []
        for key, value in parameters.items():
            # Special condition for composed fields like IgnoredScope
            if isinstance(value, IgnoredScope):
                dump_parameters(model, value.__dict__, [key])
                continue
            rt_path = ["nncf", "quantization"] + path + [key]
            model.set_rt_info(str(value), rt_path)
    except RuntimeError as e:
        nncf_logger.debug(f"Unable to dump optimization parameters due to error: {e}")


@tracked_function(NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset"])
def native_quantize_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )

    graph = GraphConverter.create_nncf_graph(model)
    quantized_model = quantization_algorithm.apply(model, graph, dataset=calibration_dataset)

    if is_weight_compression_needed(advanced_parameters):
        compress_quantize_weights_transformation(quantized_model)

    dump_parameters(
        quantized_model,
        {
            "preset": preset.value,
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
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
        model,
        calibration_dataset,
        preset,
        target_device,
        subset_size,
        fast_bias_correction,
        model_type,
        ignored_scope,
        copied_parameters,
    )

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
            advanced_accuracy_restorer_parameters.num_ranking_processes,
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
            "preset": preset.value,
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
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
    if should_use_pot(advanced_parameters):
        from nncf.openvino.pot.quantization.quantize_model import quantize_impl as pot_quantize_impl

        quantize_fn = pot_quantize_impl
    else:
        quantize_fn = native_quantize_impl

    return quantize_fn(
        model,
        calibration_dataset,
        preset,
        target_device,
        subset_size,
        fast_bias_correction,
        model_type,
        ignored_scope,
        advanced_parameters,
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
    preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
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
    if should_use_pot(advanced_quantization_parameters):
        from nncf.openvino.pot.quantization.quantize_model import (
            quantize_with_accuracy_control_impl as pot_quantize_with_accuracy_control_impl,
        )

        quantize_with_accuracy_control_fn = pot_quantize_with_accuracy_control_impl
    else:
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


def compress_weights(model: ov.Model) -> ov.Model:
    """
    Implementation of the `compress_weights()` method for the OpenVINO backend.
    """
    insert_pre_compression_operations(model)
    compress_quantize_weights_transformation(model)

    return model
