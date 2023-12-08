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
from typing import Optional, Union

import torch

from nncf.common.quantization.structs import QuantizationPreset
from nncf.data import Dataset
from nncf.parameters import CompressWeightsMode
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.algorithms.post_training.algorithm import PostTrainingQuantization
from nncf.scopes import IgnoredScope
from nncf.torch.model_creation import wrap_model
from nncf.torch.nncf_module_replacement import replace_modules_by_nncf_modules
from nncf.torch.quantization.weights_compression import insert_pre_compression_operations

DEFAULT_RANGE_TYPE = "mean_min_max"


def quantize_impl(
    model: torch.nn.Module,
    calibration_dataset: Dataset,
    preset: Union[QuantizationPreset, None],
    target_device: TargetDevice,
    subset_size: int,
    fast_bias_correction: bool,
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
        raise RuntimeError("target_device == CPU_SPR is not supported")

    copied_model = deepcopy(model)

    example_input = next(iter(calibration_dataset.get_inference_data()))
    nncf_network = wrap_model(copied_model.eval(), example_input)

    quantization_algorithm = PostTrainingQuantization(
        preset=preset,
        target_device=target_device,
        subset_size=subset_size,
        fast_bias_correction=fast_bias_correction,
        model_type=model_type,
        ignored_scope=ignored_scope,
        advanced_parameters=advanced_parameters,
    )

    quantized_model = quantization_algorithm.apply(
        nncf_network, nncf_network.nncf.get_graph(), dataset=calibration_dataset
    )

    quantized_model.nncf.disable_dynamic_graph_building()

    return quantized_model


def compress_weights_impl(
    model: torch.nn.Module,
    mode=CompressWeightsMode.INT8_ASYM,
    ratio: Optional[float] = None,
    group_size: Optional[int] = None,
    ignored_scope: Optional[IgnoredScope] = None,
) -> torch.nn.Module:
    """
    Implementation of the `compress_weights()` method for the PyTorch backend. Currently it supports INT8
    mode only with default ratio and group_size.

    :param model: a Torch model for compression.
    :param mode: Defines a mode for weight compression.
        INT8_SYM stands for 8-bit integer symmetric quantization of all weights.
            Weights are quantized symmetrically with a fixed zero point equals to 128.
        INT8_ASYM is the same as INT8_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        INT4_SYM stands for a mixed-precision weights quantization with 4-bit integer as a primary precision.
            Weights are quantized to a primary precision symmetrically with a fixed zero point equals to 8.
            All embeddings and the last layer are always compressed to a backup precision, which is INT8_ASYM,
            by default. All others are quantized whether to 4-bit integer or to a backup precision depending on
            criteria and the given ratio.
        INT4_ASYM is the same as INT4_SYM mode, but weights are quantized to a primary precision asymmetrically
            with a typical non-fixed zero point.
        NF4 is the same as INT4_SYM mode, but primary precision is NF4 data type without zero point.
    :param ratio: the ratio between baseline and backup precisions (e.g. 0.9 means 90% of layers quantized to NF4
        and the rest to INT8_ASYM).
    :param group_size: number of weights (e.g. 128) in the channel dimension that share quantization parameters (scale).
        The value -1 means no grouping.
    :param ignored_scope: An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :return: The non-trainable model with compressed weights and dequantization operations.
    """
    if ignored_scope is not None:
        raise AttributeError("Torch backend does not support ignored scope.")
    if mode != CompressWeightsMode.INT8_ASYM:
        raise AttributeError(
            f"Torch backend supports only INT8_ASYM mode for weight compression, but given {mode} mode."
        )
    compressed_model, _ = replace_modules_by_nncf_modules(model)
    insert_pre_compression_operations(model)

    return compressed_model
