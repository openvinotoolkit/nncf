"""
 Copyright (c) 2023 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from typing import Any, Callable, Iterable, Optional

import openvino.runtime as ov

from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.experimental.openvino_native.quantization.quantize import quantize_impl as native_quantize_impl
from nncf.experimental.openvino_native.quantization.quantize import \
    quantize_with_accuracy_control_impl as native_quantize_with_accuracy_control_impl
from nncf.openvino.pot.quantization.quantize import quantize_impl as pot_quantize_impl
from nncf.openvino.pot.quantization.quantize import \
    quantize_with_accuracy_control_impl as pot_quantize_with_accuracy_control_impl
from nncf.openvino.quantization.backend_parameters import BackendParameters
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.scopes import IgnoredScope

USE_POT_AS_DEFAULT = True


def should_use_pot(advanced_parameters: Optional[AdvancedQuantizationParameters]) -> bool:
    """
    Returns True if POT should be used for quantization, False otherwise.

    :param advanced_parameters: Advanced quantization parameters.
    :return: True if POT should be used, False otherwise.
    """
    if advanced_parameters is None:
        return USE_POT_AS_DEFAULT
    return advanced_parameters.backend_params.get(
        BackendParameters.USE_POT, USE_POT_AS_DEFAULT)


def quantize_impl(model: ov.Model,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType],
                  ignored_scope: Optional[IgnoredScope],
                  advanced_parameters: Optional[AdvancedQuantizationParameters]) -> ov.Model:

    """
    Implementation of the `quantize()` method for the OpenVINO backend.
    """
    if should_use_pot(advanced_parameters):
        quantize_fn = pot_quantize_impl
    else:
        quantize_fn = native_quantize_impl

    return quantize_fn(
        model, calibration_dataset, preset, target_device, subset_size,
        fast_bias_correction, model_type, ignored_scope, advanced_parameters)


def quantize_with_accuracy_control_impl(
    model: ov.Model,
    calibration_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fn: Callable[[ov.CompiledModel, Iterable[Any]], float],
    max_drop: float,
    preset: QuantizationPreset,
    target_device: TargetDevice,
    subset_size: int,
    fast_bias_correction: bool,
    model_type: Optional[ModelType],
    ignored_scope: Optional[IgnoredScope],
    advanced_quantization_parameters: Optional[AdvancedQuantizationParameters],
    advanced_accuracy_restorer_parameters: Optional[AdvancedAccuracyRestorerParameters]) -> ov.Model:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend.
    """
    if should_use_pot(advanced_quantization_parameters):
        quantize_with_accuracy_control_fn = pot_quantize_with_accuracy_control_impl
    else:
        quantize_with_accuracy_control_fn = native_quantize_with_accuracy_control_impl
    return quantize_with_accuracy_control_fn(
        model, calibration_dataset, validation_dataset, validation_fn, max_drop, preset,
        target_device, subset_size, fast_bias_correction, model_type, ignored_scope,
        advanced_quantization_parameters, advanced_accuracy_restorer_parameters)
