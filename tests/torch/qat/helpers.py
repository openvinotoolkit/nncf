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

import gc
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import torch

from examples.torch.common.example_logger import logger
from examples.torch.common.execution import start_worker
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.range_estimator import RangeEstimatorParameters
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.torch.nncf_network import NNCFNetwork
from nncf.torch.quantization.layers import BaseQuantizer


def convert_quantization_mode(mode: Optional[str]) -> QuantizationScheme:
    if mode is None:
        return None

    if mode == "symmetric":
        return QuantizationScheme.SYMMETRIC
    if mode == "asymmetric":
        return QuantizationScheme.ASYMMETRIC
    raise RuntimeError(f"Unknown quantization mode: {mode}")


def convert_quantization_params(conf: Optional[Dict[str, Any]]) -> QuantizationParameters:
    if conf is None:
        return QuantizationParameters()

    return QuantizationParameters(
        num_bits=conf.get("bits", None),
        mode=convert_quantization_mode(conf.get("mode", None)),
        signedness_to_force=conf.get("signed", None),
        per_channel=None,  # Always use the default parameter for per_channel parameters to prevent
        # accuracy degradation due to the fact that per_channel=False for activation will make
        # depthwise convolutions activations quantizers work in the per tensor mode
        # which does not make sense in case of the CPU target device.
    )


def convert_overflow_fix_param(param: Optional[str]) -> OverflowFix:
    if param is None:
        return OverflowFix.FIRST_LAYER
    if param == "enable":
        return OverflowFix.ENABLE
    if param == "disable":
        return OverflowFix.DISABLE
    if param == "first_layer_only":
        return OverflowFix.FIRST_LAYER
    raise RuntimeError(f"Overflow fix param {param} is unknown.")


def convert_quantization_preset(preset: str) -> QuantizationPreset:
    if preset == "performance":
        return QuantizationPreset.PERFORMANCE
    if preset == "mixed":
        return QuantizationPreset.MIXED
    raise RuntimeError(f"Preset {preset} is unknown.")


def get_range_init_type(config_quantization_params: Dict[str, Any]) -> RangeEstimatorParameters:
    if (
        "initializer" in config_quantization_params
        and "range" in config_quantization_params["initializer"]
        and "type" in config_quantization_params["initializer"]["range"]
    ):
        range_init_type = config_quantization_params["initializer"]["range"]["type"]
        if range_init_type == "mean_percentile":
            return RangeEstimatorParametersSet.MEAN_QUANTILE
        logger.info(f"Unknown range init type: {range_init_type}, default range init type is used.")
    return RangeEstimatorParametersSet.MINMAX


def get_quantization_preset(config_quantization_params: Dict[str, Any]) -> Optional[QuantizationPreset]:
    if "preset" not in config_quantization_params:
        return None
    return convert_quantization_preset(config_quantization_params["preset"])


def get_advanced_ptq_parameters(config_quantization_params: Dict[str, Any]) -> AdvancedQuantizationParameters:
    range_estimator_params = get_range_init_type(config_quantization_params)
    return AdvancedQuantizationParameters(
        overflow_fix=convert_overflow_fix_param(config_quantization_params.get("overflow_fix")),
        weights_quantization_params=convert_quantization_params(config_quantization_params.get("weights")),
        activations_quantization_params=convert_quantization_params(config_quantization_params.get("activations")),
        weights_range_estimator_params=range_estimator_params,
        activations_range_estimator_params=range_estimator_params,
    )


def get_num_samples(config_quantization_params: Dict[str, Any]) -> int:
    if (
        "initializer" in config_quantization_params
        and "range" in config_quantization_params["initializer"]
        and "num_init_samples" in config_quantization_params["initializer"]["range"]
    ):
        num_samples = config_quantization_params["initializer"]["range"]["num_init_samples"]
        if isinstance(num_samples, int):
            return num_samples
    return 300


def broadcast_initialized_parameters(quantized_model: NNCFNetwork):
    for module in quantized_model.modules():
        if isinstance(module, BaseQuantizer):
            module.broadcast_initialized_params()


def get_mocked_compression_ctrl():
    compression_ctrl = MagicMock()
    compression_ctrl.loss = lambda: 0.0
    compression_ctrl.statistics = lambda *args, **kwargs: []
    return compression_ctrl


def start_worker_clean_memory(*args, **kwargs):
    result = start_worker(*args, **kwargs)
    gc.collect()
    torch.cuda.empty_cache()
    return result
