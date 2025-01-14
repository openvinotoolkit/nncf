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

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypeVar, Union

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.common.factory import NNCFGraphFactory
from nncf.common.factory import StatisticsAggregatorFactory
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.openvino.graph.metatypes.groups import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS
from nncf.openvino.graph.metatypes.openvino_metatypes import OVIfMetatype
from nncf.openvino.graph.metatypes.openvino_metatypes import get_node_metatype
from nncf.openvino.graph.model_utils import remove_friendly_name_duplicates
from nncf.openvino.graph.nncf_graph_builder import GraphConverter
from nncf.openvino.graph.node_utils import get_number_if_op
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.openvino.quantization.backend_parameters import is_weight_compression_needed
from nncf.openvino.quantization.quantize_ifmodel import apply_algorithm_if_bodies
from nncf.openvino.rt_info import dump_parameters
from nncf.parameters import BackupMode
from nncf.parameters import CompressWeightsMode
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import SensitivityMetric
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import convert_to_dict_recursively
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import calculate_accuracy_drop
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.quantize_model import BATCHWISE_STATISTICS_WARNING
from nncf.quantization.quantize_model import is_model_no_batchwise_support
from nncf.quantization.quantize_model import quantize_with_tune_hyperparams
from nncf.quantization.quantize_model import warning_model_no_batchwise_support
from nncf.quantization.statistics_caching import cache_weight_compression_statistics
from nncf.quantization.statistics_caching import register_statistics_for_algorithm
from nncf.scopes import IgnoredScope
from nncf.scopes import validate_ignored_scope

TTensor = TypeVar("TTensor")


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
    graphs = {}

    def _extract_all_subgraphs(model: ov.Model, current_id: str) -> None:
        """
        Creates all inner subgraphs from If nodes and adds them to 'graphs'.

        :param model: Model.
        :param current_id: Current graph id.
        """
        graphs[current_id] = NNCFGraphFactory.create(model)
        for op in model.get_ops():
            if get_node_metatype(op) == OVIfMetatype:
                _extract_all_subgraphs(op.get_function(0), op.get_friendly_name() + "_then")
                _extract_all_subgraphs(op.get_function(1), op.get_friendly_name() + "_else")

    main_model_graph_id = "main_model_graph"
    _extract_all_subgraphs(model, main_model_graph_id)
    if ignored_scope and ignored_scope.validate:
        validate_ignored_scope(ignored_scope, graphs.values())
        ignored_scope = IgnoredScope(
            ignored_scope.names, ignored_scope.patterns, ignored_scope.types, ignored_scope.subgraphs, validate=False
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
    for graph in graphs.values():
        if is_model_no_batchwise_support(graph, advanced_parameters, model_type, OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS):
            nncf_logger.warning(BATCHWISE_STATISTICS_WARNING)
            break
    if_ops_number = get_number_if_op(model)
    nncf_logger.info(
        f"The model consists of {if_ops_number} If node(-s) with then and else bodies. \
            Main model and all If bodies will be quantized recursively."
    )
    quantized_model, _ = apply_algorithm_if_bodies(
        quantization_algorithm, model, graphs, main_model_graph_id, calibration_dataset, subset_size, 1
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


def quantize_with_accuracy_control_impl(
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
    scale_estimation: bool,
    gptq: bool,
    lora_correction: bool,
    backup_mode: BackupMode,
    advanced_parameters: Optional[AdvancedCompressionParameters] = None,
) -> ov.Model:
    """
    Implementation of the `compress_weights()` method for the OpenVINO backend.
    """
    model = remove_friendly_name_duplicates(model)
    graph = NNCFGraphFactory.create(model)
    compression_algorithm = WeightCompression(
        mode,
        ratio,
        group_size,
        ignored_scope,
        all_layers,
        sensitivity_metric,
        awq,
        subset_size,
        scale_estimation,
        gptq,
        lora_correction,
        backup_mode,
        advanced_parameters,
    )

    statistics_points = None
    if advanced_parameters and advanced_parameters.statistics_path:
        # If there is no such directory, then caches statistics
        statistics_path = Path(advanced_parameters.statistics_path)
        if not statistics_path.exists():
            cache_weight_compression_statistics(model, graph, dataset, subset_size, statistics_path)
        statistics_aggregator = StatisticsAggregatorFactory.create(model, dataset)
        compression_algorithm.set_backend_entity(model)
        _, matmul_input_to_output_nodes_map = compression_algorithm.get_compression_nodes_info(graph)
        register_statistics_for_algorithm(
            statistics_aggregator,
            model,
            graph,
            compression_algorithm,
            matmul_input_to_output_nodes_map,
        )
        statistics_aggregator.load_statistics_from_dir(statistics_path)
        statistics_points = statistics_aggregator.statistic_points

    return compression_algorithm.apply(model, graph, statistics_points, dataset)
