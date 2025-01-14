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

from typing import Any, Dict, Optional

import tensorflow as tf

import nncf
from nncf.common.initialization.dataloader import NNCFDataLoader
from nncf.common.quantization.structs import QuantizationPreset
from nncf.config import NNCFConfig
from nncf.config.structures import BNAdaptationInitArgs
from nncf.config.structures import QuantizationRangeInitArgs
from nncf.data import Dataset
from nncf.data.dataset import DataProvider
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import apply_advanced_parameters_to_config
from nncf.scopes import IgnoredScope
from nncf.scopes import convert_ignored_scope_to_list
from nncf.tensorflow.helpers.model_creation import create_compressed_model

DEFAULT_RANGE_TYPE = "mean_min_max"


# TODO(alexsu52): It is a workaround and should be removed.
class CalibrationDataLoader(NNCFDataLoader):
    """
    This class wraps the nncf.Dataset.

    This is required for proper initialization of certain compression algorithms.
    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    @property
    def batch_size(self) -> int:
        data_source = getattr(self._dataset, "_data_source")

        if not hasattr(data_source, "_batch_size"):
            return 1
        batch_size = getattr(data_source, "_batch_size")
        try:
            if isinstance(batch_size, tf.Tensor):
                batch_size = batch_size.numpy()
            batch_size = int(batch_size)
        except:  # noqa: E722
            batch_size = 1
        return batch_size

    def __iter__(self):
        def transform_fn(data_item):
            return data_item, None

        return iter(DataProvider(self._dataset.get_inference_data(), transform_fn))


def _get_default_quantization_config(preset: QuantizationPreset, subset_size: int) -> Dict[str, Any]:
    """
    Returns the default quantization config

    :param preset: A preset that controls the quantization mode
        (symmetric and asymmetric). It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric
          quantization of activations.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :return: The default quantization config.
    """
    return {
        "algorithm": "quantization",
        "preset": preset.value,
        "initializer": {
            "range": {"num_init_samples": subset_size, "type": DEFAULT_RANGE_TYPE},
            "batchnorm_adaptation": {"num_bn_adaptation_samples": subset_size},
        },
        "overflow_fix": "first_layer_only",
    }


def _create_nncf_config(
    preset: QuantizationPreset,
    target_device: TargetDevice,
    subset_size: int,
    ignored_scope: Optional[IgnoredScope],
    advanced_parameters: Optional[AdvancedQuantizationParameters],
) -> NNCFConfig:
    """
    Creates the NNCFConfig for the quantization algorithm.

    :param preset: A preset that controls the quantization mode
        (symmetric and asymmetric). It can take the following values:
        - `performance`: Symmetric quantization of weights and activations.
        - `mixed`: Symmetric quantization of weights and asymmetric
          quantization of activations.
    :param target_device: A target device the specificity of which will be taken
        into account while compressing in order to obtain the best performance
        for this type of device.
    :param subset_size: Size of a subset to calculate activations
        statistics used for quantization.
    :param ignored_scope:  An ignored scope that defined the list of model control
        flow graph nodes to be ignored during quantization.
    :param advanced_parameters: Advanced quantization parameters for
        fine-tuning the quantization algorithm.
    :return: NNCFConfig for the quantization algorithm.
    """
    compression_config = _get_default_quantization_config(preset, subset_size)

    if ignored_scope is not None:
        _ignored_scope = convert_ignored_scope_to_list(ignored_scope)
        if "ignored_scopes" in compression_config:
            compression_config["ignored_scopes"].extend(_ignored_scope)
        else:
            compression_config["ignored_scopes"] = _ignored_scope
        compression_config["validate_scopes"] = ignored_scope.validate

    if advanced_parameters is not None:
        compression_config = apply_advanced_parameters_to_config(compression_config, advanced_parameters)

    return NNCFConfig({"target_device": target_device.value, "compression": compression_config})


def quantize_impl(
    model: tf.Module,
    calibration_dataset: Dataset,
    mode: Optional[QuantizationMode] = None,
    preset: Optional[QuantizationPreset] = None,
    target_device: TargetDevice = TargetDevice.ANY,
    subset_size: int = 300,
    fast_bias_correction: bool = True,
    model_type: Optional[ModelType] = None,
    ignored_scope: Optional[IgnoredScope] = None,
    advanced_parameters: Optional[AdvancedQuantizationParameters] = None,
) -> tf.Module:
    """
    Implementation of the `quantize()` method for the TensorFlow backend.
    """
    if model_type is not None:
        raise ValueError(f"model_type={model_type} is not supported")
    if fast_bias_correction is False:
        raise ValueError(f"fast_bias_correction={fast_bias_correction} is not supported")
    if ignored_scope is not None and ignored_scope.types:
        raise nncf.InternalError(
            "Quantization algorithm form the TensorFlow backend "
            "does not support operation types in the ignored "
            "scopes yet"
        )
    if target_device == TargetDevice.CPU_SPR:
        raise nncf.InternalError("target_device == CPU_SPR is not supported.")

    if mode is not None:
        raise ValueError(f"mode={mode} is not supported")

    if preset is None:
        preset = QuantizationPreset.PERFORMANCE

    nncf_config = _create_nncf_config(preset, target_device, subset_size, ignored_scope, advanced_parameters)

    calibration_data_loader = CalibrationDataLoader(calibration_dataset)
    nncf_config.register_extra_structs(
        [
            QuantizationRangeInitArgs(data_loader=calibration_data_loader),
            BNAdaptationInitArgs(data_loader=calibration_data_loader),
        ]
    )

    _, compressed_model = create_compressed_model(model=model, config=nncf_config)

    return compressed_model
