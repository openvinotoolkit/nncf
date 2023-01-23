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

from typing import Optional

import onnx

from nncf.data import Dataset
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.parameters import convert_ignored_scope_to_list
from nncf.parameters import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.common.quantization.structs import QuantizationPreset
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm  import PostTrainingQuantizationParameters
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_ONNX_CATEGORY


@tracked_function(NNCF_ONNX_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset"])
def quantize_impl(model: onnx.ModelProto,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType] = None,
                  ignored_scope: Optional[IgnoredScope] = None) -> onnx.ModelProto:
    """
    Implementation of the `quantize()` method for the ONNX backend.
    """
    if model_type is not None:
        raise ValueError(f'model_type={model_type} is not supported')
    if ignored_scope is not None and ignored_scope.types is not None:
        raise RuntimeError('Quantization algorithm from the ONNX backend '
                           'does not support operation types in the ignored '
                           'scopes yet')
    if target_device == TargetDevice.CPU_SPR:
        raise RuntimeError('target_device == CPU_SPR is not supported.')

    quantization_parameters = PostTrainingQuantizationParameters(
        preset=preset,
        target_device=target_device,
        number_samples=subset_size,
        ignored_scopes=convert_ignored_scope_to_list(ignored_scope),
        fast_bias_correction=fast_bias_correction
    )

    quantization_algorithm = PostTrainingQuantization(quantization_parameters)
    quantized_model = quantization_algorithm.apply(model, dataset=calibration_dataset)

    return quantized_model
