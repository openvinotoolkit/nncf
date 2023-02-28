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

import openvino.runtime as ov
from openvino._offline_transformations import compress_quantize_weights_transformation

from nncf.data import Dataset
from nncf.common.quantization.structs import QuantizationPreset
from nncf.scopes import IgnoredScope
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm  import PostTrainingQuantizationParameters
from nncf.quantization.telemetry_extractors import CompressionStartedWithQuantizeApi
from nncf.telemetry import tracked_function
from nncf.telemetry.events import NNCF_OV_CATEGORY


@tracked_function(NNCF_OV_CATEGORY, [CompressionStartedWithQuantizeApi(), "target_device", "preset"])
def quantize_impl(model: ov.Model,
                  calibration_dataset: Dataset,
                  preset: QuantizationPreset,
                  target_device: TargetDevice,
                  subset_size: int,
                  fast_bias_correction: bool,
                  model_type: Optional[ModelType] = None,
                  ignored_scope: Optional[IgnoredScope] = None) -> ov.Model:
    """
    Implementation of the `quantize()` method for the OpenVINO backend via the OpenVINO Runtime API.
    """
    if model_type is not None:
        raise ValueError(f'model_type={model_type} is not supported')

    quantization_parameters = PostTrainingQuantizationParameters(
        preset=preset,
        target_device=target_device,
        number_samples=subset_size,
        ignored_scopes=ignored_scope,
        fast_bias_correction=fast_bias_correction
    )

    quantization_algorithm = PostTrainingQuantization(quantization_parameters)
    quantized_model = quantization_algorithm.apply(model, dataset=calibration_dataset)
    compress_quantize_weights_transformation(quantized_model)

    return quantized_model
