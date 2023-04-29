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

from copy import deepcopy
from typing import Any, Callable, Iterable, Optional

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.utils.backend import get_backend
from nncf.common.utils.timer import timer
from nncf.data.dataset import Dataset
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.openvino.quantization.backend_parameters import is_weight_compression_needed
from nncf.openvino.quantization.quantize_model import quantize_impl
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.quantization.algorithms.accuracy_control.algorithm import get_algo_backend
from nncf.scopes import IgnoredScope


def _match_const_nodes_names(initial_model: ov.Model, quantized_model: ov.Model) -> None:
    """
    Replaces the name of the constant node in the `quantized_model`
    with the name of the corresponding constant node in the `initial_model`.

    :param initial_model: Initial model.
    :param quantized_model_graph: Quantized model.
    """
    initial_name_to_const_map = {
        op.get_friendly_name(): op for op in initial_model.get_ops() if op.get_type_name() == "Constant"
    }
    modified_name_to_const_map = {
        op.get_friendly_name(): op for op in quantized_model.get_ops() if op.get_type_name() == "Constant"
    }

    for initial_name in initial_name_to_const_map:
        num_matches = 0

        name_to_search = initial_name
        if "compressed" in name_to_search:
            name_to_search = name_to_search[: name_to_search.rfind("compressed") - 1]

        for modified_name, const_op in modified_name_to_const_map.items():
            if modified_name.startswith(name_to_search):
                num_matches += 1
                const_op.set_friendly_name(initial_name)

        if num_matches != 1:
            raise RuntimeError(
                "Unexpected Behavior: number of matches greater than 1\n"
                f"num_matches: {num_matches}, name: {initial_name}"
            )


def quantize_with_accuracy_control(
    model: ov.Model,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[Any, Iterable[Any]], float],
    max_drop: float = 0.01,
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
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend via POT.
    """
    if advanced_accuracy_restorer_parameters is None:
        advanced_accuracy_restorer_parameters = AdvancedAccuracyRestorerParameters()

    if advanced_accuracy_restorer_parameters.tune_hyperparams:
        raise RuntimeError(
            "Quantization algorithm with accuracy control from the "
            "OpenVINO backend does not support tuning hyperparams yet"
        )
    if advanced_accuracy_restorer_parameters.convert_to_mixed_preset:
        raise RuntimeError(
            "Quantization algorithm with accuracy control from the "
            "OpenVINO backend does not support the option to convert "
            "to the mixed preset yet"
        )

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

    # We need to match constant names when the
    # quantized model was got using POT. For example, we have the
    # `Constant_63974886249` constant name in the quantized model,
    # but `Constant_6397` in the initial model.
    # The `_collect_original_biases_and_weights()`` method throws
    # the error otherwise.
    _match_const_nodes_names(model, quantized_model)

    backend = get_backend(model)
    algo_backend = get_algo_backend(backend)

    nncf_logger.info("Validation of initial model was started")
    with timer():
        initial_metric = validation_fn(algo_backend.prepare_for_inference(model), validation_dataset.get_data())
    nncf_logger.info(f"Metric of initial model: {initial_metric}")

    nncf_logger.info("Validation of quantized model was started")
    with timer():
        quantized_metric = validation_fn(
            algo_backend.prepare_for_inference(quantized_model), validation_dataset.get_data()
        )
    nncf_logger.info(f"Metric of quantized model: {quantized_metric}")

    ranking_subset_size = subset_size
    if advanced_accuracy_restorer_parameters.ranking_subset_size is not None:
        ranking_subset_size = advanced_accuracy_restorer_parameters.ranking_subset_size

    accuracy_aware_loop = QuantizationAccuracyRestorer(
        ranking_subset_size=ranking_subset_size,
        max_num_iterations=advanced_accuracy_restorer_parameters.max_num_iterations,
        max_drop=max_drop,
    )
    quantized_model = accuracy_aware_loop.restore_accuracy(
        model, initial_metric, quantized_model, quantized_metric, validation_dataset, validation_fn
    )
    if compress_weights:
        compress_quantize_weights_transformation(quantized_model)

    return quantized_model
