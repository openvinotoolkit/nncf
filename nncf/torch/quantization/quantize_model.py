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
from nncf.common.factory import NNCFGraphFactory
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.parameters import BackupMode
from nncf.parameters import CompressWeightsMode
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import SensitivityMetric
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.weight_compression.algorithm import WeightCompression
from nncf.quantization.quantize_model import warning_model_no_batchwise_support
from nncf.scopes import IgnoredScope
from nncf.torch.graph.operator_metatypes import OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS
from nncf.torch.model_creation import wrap_model

DEFAULT_RANGE_TYPE = "mean_min_max"


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
        raise ValueError(f"fast_bias_correction={fast_bias_correction} is not supported")
    if target_device == TargetDevice.CPU_SPR:
        raise nncf.InternalError("target_device == CPU_SPR is not supported")
    if mode is not None:
        raise ValueError(f"mode={mode} is not supported")

    copied_model = deepcopy(model)

    example_input = next(iter(calibration_dataset.get_inference_data()))
    nncf_network = wrap_model(copied_model.eval(), example_input, trace_parameters=True)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )
    graph = nncf_network.nncf.get_graph()
    warning_model_no_batchwise_support(graph, advanced_parameters, model_type, OPERATIONS_OUTPUT_HAS_NO_BATCH_AXIS)
    quantized_model = quantization_algorithm.apply(nncf_network, graph, dataset=calibration_dataset)

    quantized_model.nncf.disable_dynamic_graph_building()

    return quantized_model


def compress_weights_impl(
    model: torch.nn.Module,
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
) -> torch.nn.Module:
    """
    Implementation of the `compress_weights()` method for the PyTorch backend.
    """

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
    graph = NNCFGraphFactory.create(model)
    return compression_algorithm.apply(model, graph, dataset=dataset)
