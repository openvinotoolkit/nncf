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
from typing import Optional

import torch

import nncf
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.quantize_model import warning_model_no_batchwise_support
from nncf.scopes import IgnoredScope
from nncf.torch.function_hook import wrap_model
from nncf.torch.function_hook.nncf_graph.nncf_graph_builder import GraphModelWrapper
from nncf.torch.graph.operator_metatypes import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS


def quantize_impl(
    model: torch.nn.Module,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> torch.nn.Module:
    """
    Implementation of the `quantize()` method for the PyTorch backend.
    """
    if fast_bias_correction is False:
        msg = f"fast_bias_correction={fast_bias_correction} is not supported"
        raise ValueError(msg)
    if target_device == TargetDevice.CPU_SPR:
        msg = "target_device == CPU_SPR is not supported"
        raise nncf.InternalError(msg)
    if mode is not None:
        msg = f"mode={mode} is not supported"
        raise ValueError(msg)

    copied_model = deepcopy(model)
    example_input = next(iter(calibration_dataset.get_inference_data()))

    copied_model = wrap_model(copied_model)
    model_wrapper = GraphModelWrapper(copied_model, example_input)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )
    graph = model_wrapper.build_graph()

    warning_model_no_batchwise_support(graph, advanced_parameters, model_type, OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS)
    quantized_model = quantization_algorithm.apply(model_wrapper, graph, dataset=calibration_dataset)

    return quantized_model.model
