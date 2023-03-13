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

from typing import Callable, Any, Iterable, Optional

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.data.dataset import Dataset
from nncf.common.logging import nncf_logger
from nncf.common.utils.backend import get_backend
from nncf.common.quantization.structs import QuantizationPreset
from nncf.scopes import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.accuracy_control.algorithm import get_algo_backend
from nncf.quantization.algorithms.accuracy_control.algorithm import QuantizationAccuracyRestorer
from nncf.openvino.quantization.quantize import quantize_impl


def _match_const_nodes_names(initial_model: ov.Model, quantized_model: ov.Model) -> None:
    """
    Replaces the name of the constant node in the `quantized_model`
    with the name of the corresponding constant node in the `initial_model`.

    :param initial_model: Initial model.
    :param quantized_model_graph: Quantized model.
    """
    initial_name_to_const_map = {
        op.get_friendly_name(): op for op in initial_model.get_ops() if op.get_type_name() == 'Constant'
    }
    modified_name_to_const_map = {
        op.get_friendly_name(): op for op in quantized_model.get_ops() if op.get_type_name() == 'Constant'
    }

    for initial_name in initial_name_to_const_map:
        num_matches = 0
        for modified_name, const_op in modified_name_to_const_map.items():
            if modified_name.startswith(initial_name):
                num_matches += 1
                const_op.set_friendly_name(initial_name)
        assert num_matches == 1


def quantize_with_accuracy_control(model: ov.Model,
                                   calibration_dataset: Dataset,
                                   validation_dataset: Dataset,
                                   validation_fn: Callable[[Any, Iterable[Any]], float],
                                   max_drop: float = 0.01,
                                   preset: QuantizationPreset = QuantizationPreset.PERFORMANCE,
                                   target_device: TargetDevice = TargetDevice.ANY,
                                   subset_size: int = 300,
                                   fast_bias_correction: bool = True,
                                   model_type: Optional[ModelType] = None,
                                   ignored_scope: Optional[IgnoredScope] = None) -> ov.Model:
    """
    Implementation of the `quantize_with_accuracy_control()` method for the OpenVINO backend via POT.
    """
    quantized_model = quantize_impl(model, calibration_dataset, preset, target_device, subset_size,
                                    fast_bias_correction, model_type, ignored_scope, compress_weights=False)

    # We need to match constant names when the
    # quantized model was got using POT. For example, we have the
    # `Constant_63974886249` constant name in the quantized model,
    # but `Constant_6397` in the initial model.
    # The `_collect_original_biases_and_weights()`` method throws
    # the error otherwise.
    _match_const_nodes_names(model, quantized_model)

    backend = get_backend(model)
    algo_backend = get_algo_backend(backend)

    initial_metric = validation_fn(algo_backend.prepare_for_inference(model),
                                   validation_dataset.get_data())
    nncf_logger.info(f'Metric of initial model: {initial_metric}')

    quantized_metric = validation_fn(algo_backend.prepare_for_inference(quantized_model),
                                     validation_dataset.get_data())
    nncf_logger.info(f'Metric of quantized model: {quantized_metric}')

    accuracy_aware_loop = QuantizationAccuracyRestorer(algo_backend, max_drop=max_drop, is_native=False)
    quantized_model = accuracy_aware_loop.restore_accuracy(model, initial_metric,
                                                           quantized_model, quantized_metric,
                                                           validation_dataset, validation_fn)
    compress_quantize_weights_transformation(quantized_model)

    return quantized_model
