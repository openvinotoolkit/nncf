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

import torch

from nncf.common.logging import nncf_logger
from nncf.common.quantization.structs import QuantizationPreset
from nncf.config import NNCFConfig
from nncf.data import Dataset
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.algorithms.bias_correction.algorithm import BiasCorrection
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantizationParameters
from nncf.scopes import IgnoredScope
from nncf.torch.model_creation import create_nncf_network


def quantize_impl(
    model: torch.nn.Module,
    calibration_dataset: Dataset,
    preset: QuantizationPreset,
    target_device: TargetDevice,
    subset_size: int,
    fast_bias_correction: bool,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
) -> torch.nn.Module:
    """
    Experimental implementation of the `quantize()` method for the PyTorch backend.
    """

    dataset_iter = iter(calibration_dataset.get_inference_data())
    input_shape = tuple(next(dataset_iter).shape)
    nncf_config = NNCFConfig({'input_info': {'sample_size': input_shape}})
    model.eval()
    nncf_network = create_nncf_network(model, nncf_config)
    params = PostTrainingQuantizationParameters(
        number_samples=subset_size,
        preset=preset,
        target_device=target_device,
        ignored_scopes=ignored_scope,
        model_type=model_type,
        fast_bias_correction=fast_bias_correction,
    )

    nncf_logger.warning('BiasCorrection algorithm is not supported by Torch backend by now.')
    params.algorithms.pop(BiasCorrection, None)

    quantization_algorithm = PostTrainingQuantization(params)

    quantized_model = quantization_algorithm.apply(nncf_network, dataset=calibration_dataset)
    return quantized_model
