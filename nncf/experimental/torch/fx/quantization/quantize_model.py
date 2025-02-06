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
import torch.fx
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch.ao.quantization.pt2e.qat_utils import _fold_conv_bn_qat
from torch.ao.quantization.pt2e.utils import _disallow_eval_train
from torch.fx import GraphModule
from torch.fx.passes.infra.pass_manager import PassManager

import nncf
from nncf.common.factory import NNCFGraphFactory
from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.experimental.torch.fx.quantization.backend_parameters import is_weight_compression_needed
from nncf.experimental.torch.fx.transformations import apply_quantization_transformations
from nncf.experimental.torch.fx.transformations import compress_post_quantize_transformation
from nncf.experimental.torch.fx.transformations import fq_weights_transformation
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
from nncf.scopes import IgnoredScope

DEFAULT_RANGE_TYPE = "mean_min_max"


def quantize_impl(
    model: torch.fx.GraphModule,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> torch.fx.GraphModule:
    """
    Implementation of the `quantize()` method for the Torch FX backend.
    """
    nncf_logger.warning(
        "Experimental Torch FX quantization backend is being used for the given torch.fx.GraphModule model."
        " Torch FX PTQ is an experimental feature, consider using Torch or OpenVino PTQ backends"
        " in case of errors or a poor model performance."
    )
    if target_device == TargetDevice.CPU_SPR:
        raise nncf.InternalError("target_device == CPU_SPR is not supported")
    if mode is not None:
        raise ValueError(f"mode={mode} is not supported")

    original_graph_meta = model.meta

    copied_model = deepcopy(model)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )

    # To make it easier for bias correction algorithms.
    apply_quantization_transformations(copied_model)

    nncf_graph = NNCFGraphFactory.create(copied_model)
    quantized_model = quantization_algorithm.apply(copied_model, nncf_graph, dataset=calibration_dataset)

    if is_weight_compression_needed(advanced_parameters):
        compress_post_quantize_transformation(quantized_model)
    else:
        fq_weights_transformation(quantized_model)

    # Magic. Without this call compiled model
    # is not preformant
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    quantized_model = _fold_conv_bn_qat(quantized_model)
    pm = PassManager([DuplicateDQPass()])

    quantized_model = pm(quantized_model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    quantized_model = pm(quantized_model).graph_module

    quantized_model.meta.update(original_graph_meta)
    quantized_model = _disallow_eval_train(quantized_model)
    # Each transformation adds a duplicate tensor value to the model buffer.
    #  This step removes the duplicates tensor values from the buffer.
    quantized_model = GraphModule(quantized_model, quantized_model.graph)

    return quantized_model


def compress_weights_impl(
    model: torch.fx.GraphModule,
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
) -> torch.fx.GraphModule:
    """
    Implementation of the `compress_weights()` method for the Torch Fx backend.
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
    compressed_model = compression_algorithm.apply(model, graph, dataset=dataset)
    compressed_model = GraphModule(compressed_model, compressed_model.graph)
    compressed_model = _disallow_eval_train(compressed_model)

    return compressed_model
