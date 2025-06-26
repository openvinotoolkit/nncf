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

import sys
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar, Union

import onnx
from onnx.external_data_helper import ExternalDataInfo
from onnx.external_data_helper import _get_all_tensors
from onnx.external_data_helper import load_external_data_for_model
from onnx.external_data_helper import uses_external_data

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging.logger import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.onnx.graph.metatypes.groups import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS
from nncf.onnx.graph.model_metadata import MetadataKey
from nncf.onnx.graph.model_metadata import remove_metadata
from nncf.onnx.graph.model_metadata import set_metadata
from nncf.onnx.graph.nncf_graph_builder import GraphConverter
from nncf.onnx.graph.passes import apply_preprocess_passes
from nncf.onnx.quantization.backend_parameters import get_external_data_dir
from nncf.parameters import BackupMode
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import SensitivityMetric
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import calculate_accuracy_drop
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.quantize_model import quantize_with_tune_hyperparams
from nncf.quantization.quantize_model import warning_model_no_batchwise_support
from nncf.scopes import IgnoredScope

TTensor = TypeVar("TTensor")


def check_model_protobuf_size(model: onnx.ModelProto) -> None:
    """
    Checks whether the serialized ONNX model exceeds the 2GB protobuf size limit.
    If the size limit is exceeded, a `nncf.ValidationError` is raised.

    :param model: The ONNX model to be checked.
    """
    maximum_protobuf = 2147483648  # Limitation of single protobuf file is 2GB
    protobuf_string = model.SerializeToString()
    if sys.getsizeof(protobuf_string) > maximum_protobuf:
        msg = (
            "The protobuf of onnx model is too large (>2GB). "
            "Please load the model with the `load_external_data` flag set to `False`. "
            "For more details, please visit: https://onnx.ai/onnx/repo-docs/ExternalData.html"
        )
        raise nncf.ValidationError(msg)


def check_external_data_location(model: onnx.ModelProto, external_data_dir: Optional[str] = None) -> Optional[str]:
    """
    Raises `nncf.ValidationError` if any referenced external data file does not exist, is not a regular file,
    or is a symlink.

    :param model: The ONNX model to validate.
    :param external_data_dir: Path to the directory where the external data files are expected to be located.
        If None, the current working directory is used.
    :return: Path to the directory where external data files are located, if the model uses external data.
        Returns None if no external data is used.
    """
    # If external_data_dir is not provided, we should test against the current working directory.
    external_data_dir = Path.cwd() if external_data_dir is None else Path(external_data_dir)

    if not external_data_dir.is_absolute():
        msg = (
            f"BackendParameters.EXTERNAL_DATA_DIR should be an absolute path, but {str(external_data_dir)} "
            "was provided instead."
        )
        raise nncf.ValidationError(msg)

    data_paths = set()
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            info = ExternalDataInfo(tensor)
            # `info.location` field stores file path relative to the filesystem directory where
            # the ONNX protobuf model was stored. Note, that up-directory path components such
            # .. are disallowed and should be stripped when parsing.
            # Source: https://onnx.ai/onnx/repo-docs/ExternalData.html
            external_data_file_name = Path(info.location).name  # Extract only the filename
            data_path = external_data_dir / external_data_file_name
            data_paths.add(data_path)
    for data_path in data_paths:
        if not data_path.exists() or not data_path.is_file() or data_path.is_symlink():
            msg = (
                f"Data of TensorProto (tensor name: {tensor.name}) should be stored in {str(data_path)}, "
                "but it doesn't exist or is not accessible."
            )
            raise nncf.ValidationError(msg)

    # If len(data_path) == 0, it means there are no tensors that use external data.
    return str(external_data_dir) if data_paths else None


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

    check_model_protobuf_size(model)
    external_data_dir = get_external_data_dir(advanced_parameters)
    external_data_dir = check_external_data_location(model, external_data_dir)
    if external_data_dir:
        set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, external_data_dir)
    model = apply_preprocess_passes(model)

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

    if external_data_dir:
        remove_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
        remove_metadata(quantized_model, MetadataKey.EXTERNAL_DATA_DIR)
        load_external_data_for_model(quantized_model, external_data_dir)

    return quantized_model


def quantize_with_accuracy_control_impl(
    model: onnx.ModelProto,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], tuple[float, Union[None, list[float], list[list[TTensor]]]]],
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


def compress_weights_impl(
    model: onnx.ModelProto,
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
    compression_format: CompressionFormat,
    advanced_parameters: Optional[AdvancedCompressionParameters] = None,
) -> onnx.ModelProto:
    if model.opset_import[0].version < 13:
        msg = "ONNX models with opset version < 13 do not support per-channel quantization."
        raise nncf.ValidationError(msg)

    external_data_dir = get_external_data_dir(advanced_parameters)
    external_data_dir = check_external_data_location(model, external_data_dir)
    if external_data_dir:
        set_metadata(model, MetadataKey.EXTERNAL_DATA_DIR, external_data_dir)

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
        compression_format,
        advanced_parameters,
    )
    graph = NNCFGraphFactory.create(model)

    compressed_model = compression_algorithm.apply(model, graph, dataset=dataset)

    if external_data_dir:
        remove_metadata(model, MetadataKey.EXTERNAL_DATA_DIR)
        remove_metadata(compressed_model, MetadataKey.EXTERNAL_DATA_DIR)
        load_external_data_for_model(compressed_model, external_data_dir)

    return compressed_model
