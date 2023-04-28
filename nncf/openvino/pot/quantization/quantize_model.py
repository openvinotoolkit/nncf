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

import logging
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation
from openvino.tools import pot

from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.openvino.pot.engine import OVEngine
from nncf.openvino.pot.quantization.accuracy_aware import NMSEBasedAccuracyAware
from nncf.openvino.pot.telemetry_extractors import POTImplementation
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.openvino.quantization.backend_parameters import is_weight_compression_needed
from nncf.parameters import DropType
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import StatisticsCollectorParameters
from nncf.quantization.range_estimator import StatisticsType
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeWithAccuracyControlApi
from nncf.scopes import IgnoredScope
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_OV_CATEGORY


def _convert_openvino_model_to_compressed_model(
    model: ov.Model, target_device: str
) -> pot.graph.nx_model.CompressedModel:
    """
    Serializes the provided OpenVINO model and loads the model in the POT representation.

    :param model: The OpenVINO model.
    :param target_device: The target device.
    :return: The POT representation of the provided model.
    """
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        xml_path = str(Path(tmp_dir) / "model.xml")
        bin_path = str(Path(tmp_dir) / "model.bin")
        ov.serialize(model, xml_path, bin_path)
        model_config = {
            "model_name": "model",
            "model": xml_path,
            "weights": bin_path,
        }
        pot_model = pot.load_model(model_config, target_device)

    return pot_model


def _convert_compressed_model_to_openvino_model(model: pot.graph.nx_model.CompressedModel) -> ov.Model:
    """
    Saves the provided POT compressed model and loads it as `openvino.runtime.Model` object.

    :param model: The POT compressed model.
    :return: The `openvino.runtime.Model`  object which represents the provided model.
    """
    with tempfile.TemporaryDirectory(dir=tempfile.gettempdir()) as tmp_dir:
        paths = pot.save_model(model, save_path=tmp_dir, model_name="model")
        xml_path = paths[0]["model"]
        bin_path = paths[0]["weights"]
        ie = ov.Core()
        ov_model = ie.read_model(xml_path, bin_path)
    return ov_model


def _create_ignored_scope_config(ignored_scope: Optional[IgnoredScope]) -> Dict[str, Any]:
    """
    Maps the content of `IgnoredScope` class to the `ignored` section of POT config.

    :param ignored_scope: The ignored scope
    :return: A POT ignored scope configuration as dict
    """
    if ignored_scope is None:
        return {}

    ignored = {}
    if ignored_scope.names is not None:
        ignored["scope"] = ignored_scope.names
    if ignored_scope.patterns:
        raise RuntimeError(
            "Quantization algorithm from the OpenVINO backend "
            "does not support regular expressions in the ignored "
            "scopes yet"
        )
    if ignored_scope.types is not None:
        ignored["operations"] = [{"type": type} for type in ignored_scope.types]
    return ignored


def _create_statistics_collector_config(statistics_collector_params: StatisticsCollectorParameters) -> Dict[str, Any]:
    """
    Creates a statistic collector configuration.

    :param statistics_collector_params: Statistic collector parameters
    :return: A POT statistic collector configuration as dict.
    """
    config = {}
    if statistics_collector_params.statistics_type is not None:
        config["type"] = statistics_collector_params.statistics_type.value
    if statistics_collector_params.aggregator_type is not None:
        config["aggregator"] = statistics_collector_params.aggregator_type.value
    if statistics_collector_params.clipping_value is not None:
        config["clipping_value"] = statistics_collector_params.clipping_value
    if statistics_collector_params.statistics_type in [StatisticsType.QUANTILE, StatisticsType.ABS_QUANTILE]:
        config["outlier_prob"] = statistics_collector_params.quantile_outlier_prob

    return config


def _create_range_estimator_config(range_estimator_params: RangeEstimatorParameters) -> Dict[str, Any]:
    """
    Creates a range estimator configuration.

    :param range_estimator_params: Range estimator parameters.
    :return: A POT range estimator configuration as dict.
    """
    config = {}
    min_config = _create_statistics_collector_config(range_estimator_params.min)
    if min_config:
        config["min"] = min_config

    max_config = _create_statistics_collector_config(range_estimator_params.max)
    if max_config:
        config["max"] = max_config

    return config


def _create_quantization_group_config(
    quantization_params: QuantizationParameters,
    range_estimator_params: RangeEstimatorParameters,
    backend_params: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Creates a configuration for a quantization group such as activations or weights.

    :param quantization_params: Quantization parameters.
    :param backend_params: Backend specific parameters.
    :return: A POT quantization group configuration as dict.
    """
    config = {}
    if quantization_params.num_bits is not None:
        config["bits"] = quantization_params.num_bits

    if quantization_params.mode is not None:
        config["mode"] = str(quantization_params.mode)
    if quantization_params.per_channel is not None:
        config["perchannel"] = quantization_params.per_channel

    not_supported_params = {
        "narrow_range": quantization_params.narrow_range,
        "signedness_to_force": quantization_params.signedness_to_force,
    }
    for name, value in not_supported_params.items():
        if value is not None:
            raise RuntimeError(
                "Quantization algorithm from the OpenVINO backend does not support "
                f"{name} directly, please, use backend specific parameters level_low "
                "and level_high to specify the quantization levels for activations "
                "and weights quantization groups to specify the quantization levels."
                'Example:\n {"activations" : {"level_low": 0, "level_high": 255}}\n'
                '{"weights" : {"level_low": -127, "level_high": 127}}'
            )
    if BackendParameters.LEVEL_LOW in backend_params:
        config["level_low"] = backend_params[BackendParameters.LEVEL_LOW]
    if BackendParameters.LEVEL_HIGH in backend_params:
        config["level_high"] = backend_params[BackendParameters.LEVEL_HIGH]
    config.update(_create_range_estimator_config(range_estimator_params))

    return config


def _create_quantization_config(
    preset: QuantizationPreset,
    target_device: TargetDevice,
    subset_size: int,
    fast_bias_correction: bool,
    model_type: Optional[ModelType],
    ignored_scope: Optional[IgnoredScope],
    advanced_parameters: Optional[AdvancedQuantizationParameters],
) -> Dict[str, Any]:
    """
    Creates a quantization configuration.

    :param preset: A preset that controls the quantization mode
        (symmetric and asymmetric). It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric
          quantization of activations.
    :param target_device: A target device the specificity of which will be
        taken into account while compressing in order to obtain the best
        performance for this type of device.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param fast_bias_correction: Setting this option to `False` enables
        a different bias correction method which is more accurate, in general,
        and takes more time but requires less memory.
    :param model_type: Model type is needed to specify additional patterns
        in the model. Supported only `transformer` now.
    :param ignored_scope: An ignored scope that defined the list of model
        control flow graph nodes to be ignored during quantization.
    :param advanced_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm.
    :return: A POT quantization configuration as dict.
    """
    config = {
        "target_device": target_device.value,
        "preset": preset.value,
        "stat_subset_size": subset_size,
        "use_fast_bias": fast_bias_correction,
    }

    if model_type is not None:
        config["model_type"] = model_type.value
    if ignored_scope is not None:
        config["ignored"] = _create_ignored_scope_config(ignored_scope)

    if advanced_parameters is None:
        return config

    backend_activations_parameters = advanced_parameters.backend_params.get(BackendParameters.ACTIVATIONS, {})
    activations_config = _create_quantization_group_config(
        advanced_parameters.activations_quantization_params,
        advanced_parameters.activations_range_estimator_params,
        backend_activations_parameters,
    )
    if activations_config:
        config["activations"] = activations_config

    backend_weights_parameters = advanced_parameters.backend_params.get(BackendParameters.WEIGHTS, {})
    weights_config = _create_quantization_group_config(
        advanced_parameters.weights_quantization_params,
        advanced_parameters.weights_range_estimator_params,
        backend_weights_parameters,
    )
    if weights_config:
        config["weights"] = weights_config

    if advanced_parameters.overflow_fix == OverflowFix.ENABLE:
        config["saturation_fix"] = "all"
    elif advanced_parameters.overflow_fix == OverflowFix.FIRST_LAYER:
        config["saturation_fix"] = "first_layer"
    elif advanced_parameters.overflow_fix == OverflowFix.DISABLE:
        config["saturation_fix"] = "no"

    config["inplace_statistics"] = advanced_parameters.inplace_statistics

    if advanced_parameters.quantize_outputs:
        raise RuntimeError("Quantization algorithm from the OpenVINO backend does not support output quantization yet")

    bias_correction_params = advanced_parameters.bias_correction_params
    if bias_correction_params.apply_for_all_nodes is not None:
        config["apply_for_all_nodes"] = bias_correction_params.apply_for_all_nodes
    if bias_correction_params.threshold is not None:
        config["threshold"] = bias_correction_params.threshold

    return config


def _create_engine_config(
    device: str,
    default_stat_requests_number: int,
    default_eval_requests_number: int,
    advanced_parameters: AdvancedQuantizationParameters,
) -> Dict[str, Any]:
    """
    Creates a POT engine configuration.

    :param device: A target device.
    :param stat_requests_number: The default number of infer requests that are used
        to collect statistics.
    :param default_eval_requests_number: The default number of infer requests that
        are used for model evaluation.
    :param advanced_parameters: Advanced quantization parameters.
    :return: A POT engine configuration as dict.
    """
    engine_config = {
        "device": device,
        "stat_requests_number": default_stat_requests_number,
        "eval_requests_number": default_eval_requests_number,
    }

    advanced_backend_params = [
        BackendParameters.STAT_REQUESTS_NUMBER,
        BackendParameters.EVAL_REQUESTS_NUMBER,
    ]

    for param_name in advanced_backend_params:
        param_value = advanced_parameters.backend_params.get(param_name)
        if param_value is not None:
            engine_config[param_name] = param_value

    return engine_config


@tracked_function(
    NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), POTImplementation(), "target_device", "preset"]
)
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
    pot.utils.logger.init_logger(level=logging.getLevelName(nncf_logger.getEffectiveLevel()))

    if advanced_parameters is None:
        advanced_parameters = AdvancedQuantizationParameters()

    algorithm_parameters = _create_quantization_config(
        preset, target_device, subset_size, fast_bias_correction, model_type, ignored_scope, advanced_parameters
    )

    if advanced_parameters.disable_bias_correction:
        algorithms = [
            {"name": "ActivationChannelAlignment", "params": algorithm_parameters},
            {"name": "MinMaxQuantization", "params": algorithm_parameters},
        ]
    else:
        algorithms = [{"name": "DefaultQuantization", "params": algorithm_parameters}]

    pot_model = _convert_openvino_model_to_compressed_model(model, target_device)

    engine_config = _create_engine_config(
        device="CPU",
        default_stat_requests_number=2,
        default_eval_requests_number=2,
        advanced_parameters=advanced_parameters,
    )

    engine = OVEngine(engine_config, calibration_dataset, calibration_dataset)
    pipeline = pot.create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(pot_model)
    quantized_model = _convert_compressed_model_to_openvino_model(compressed_model)

    if is_weight_compression_needed(advanced_parameters):
        compress_quantize_weights_transformation(quantized_model)

    return quantized_model


def _create_accuracy_restorer_config(
    max_drop: float, drop_type: DropType, advanced_parameters: Optional[AdvancedAccuracyRestorerParameters]
) -> Dict[str, Any]:
    """
    Creates a accuracy restorer configuration.

    :param max_drop: The maximum accuracy drop that should be achieved after
        the quantization.
    :param drop_type: The accuracy drop type, which determines how the maximum accuracy
        drop between the original model and the compressed model is calculated.
    :param advanced_parameters: Advanced parameters for fine-tuning the accuracy
        restorer algorithm.
    :return: A POT accuracy restorer configuration as dict.
    """
    config = {
        "maximal_drop": max_drop,
        "drop_type": drop_type.value,
        "metric_subset_ratio": 0.5,
    }

    if advanced_parameters is None:
        return config

    config["max_num_iterations"] = advanced_parameters.max_num_iterations
    config["tune_hyperparams"] = advanced_parameters.tune_hyperparams
    config["convert_to_mixed_preset"] = advanced_parameters.convert_to_mixed_preset
    if advanced_parameters.ranking_subset_size is not None:
        config["ranking_subset_size"] = advanced_parameters.ranking_subset_size

    return config


@tracked_function(
    NNCF_OV_CATEGORY,
    [
        CompressionStartedWithQuantizeWithAccuracyControlApi(),
        POTImplementation(),
        "target_device",
        "preset",
        "max_drop",
        "drop_type",
    ],
)
def quantize_with_accuracy_control_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[ov.CompiledModel, Iterable[Any]], float],
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
    pot.utils.logger.init_logger(level=logging.getLevelName(nncf_logger.getEffectiveLevel()))

    if advanced_quantization_parameters is None:
        advanced_quantization_parameters = AdvancedQuantizationParameters()

    if advanced_quantization_parameters.disable_bias_correction:
        raise ValueError(
            "Quantization algorithm with accuracy controll from the OpenVINO backend "
            "does not support disabling bias correction algorithm yet"
        )

    pot_model = _convert_openvino_model_to_compressed_model(model, target_device)

    engine_config = _create_engine_config(
        device="CPU",
        default_stat_requests_number=1,
        default_eval_requests_number=1,
        advanced_parameters=advanced_quantization_parameters,
    )

    # Check whether it is possible to calculate the metric for one data item.
    # pylint: disable=W0703
    use_original_metric = True
    try:
        ie = ov.Core()
        compiled_model = ie.compile_model(model, device_name="CPU")
        _ = validation_fn(compiled_model, validation_dataset.get_data(indices=[0]))
    except Exception:
        use_original_metric = False
    compression_algorithms = pot.algorithms.algorithm_selector.COMPRESSION_ALGORITHMS
    if "NMSEBasedAccuracyAware" not in compression_algorithms.registry_dict:
        compression_algorithms.register("NMSEBasedAccuracyAware")(NMSEBasedAccuracyAware)

    algotrithm_parameters = _create_accuracy_restorer_config(max_drop, drop_type, advanced_accuracy_restorer_parameters)

    algotrithm_parameters.update(
        _create_quantization_config(
            target_device,
            preset,
            subset_size,
            fast_bias_correction,
            model_type,
            ignored_scope,
            advanced_quantization_parameters,
        )
    )

    algorithms = [{"name": "NMSEBasedAccuracyAware", "params": algotrithm_parameters}]

    engine = OVEngine(engine_config, calibration_dataset, validation_dataset, validation_fn, use_original_metric)
    pipeline = pot.create_pipeline(algorithms, engine)
    compressed_model = pipeline.run(pot_model)
    quantized_model = _convert_compressed_model_to_openvino_model(compressed_model)

    if is_weight_compression_needed(advanced_quantization_parameters):
        compress_quantize_weights_transformation(quantized_model)

    return quantized_model
