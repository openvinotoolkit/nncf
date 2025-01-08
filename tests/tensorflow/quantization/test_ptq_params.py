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


import pytest

from nncf import NNCFConfig
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.advanced_parameters import apply_advanced_parameters_to_config
from nncf.quantization.range_estimator import RangeEstimatorParametersSet
from nncf.scopes import IgnoredScope
from nncf.tensorflow.quantization.quantize_model import _create_nncf_config
from nncf.tensorflow.quantization.quantize_model import _get_default_quantization_config


@pytest.mark.parametrize(
    "params",
    (
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 1,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE, quantize_outputs=True, disable_bias_correction=True
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 2,
            "ignored_scope": None,
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.ENABLE, quantize_outputs=False, disable_bias_correction=False
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 3,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.FIRST_LAYER, quantize_outputs=True, disable_bias_correction=False
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 4,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.FIRST_LAYER,
                quantize_outputs=True,
                disable_bias_correction=False,
                activations_quantization_params=QuantizationParameters(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                activations_range_estimator_params=RangeEstimatorParametersSet.MEAN_MINMAX,
                weights_quantization_params=QuantizationParameters(num_bits=8, mode=QuantizationMode.SYMMETRIC),
                weights_range_estimator_params=RangeEstimatorParametersSet.MEAN_MINMAX,
            ),
        },
    ),
)
def test_create_nncf_config(params):
    config = _create_nncf_config(**params)

    assert config["compression"]["overflow_fix"] == params["advanced_parameters"].overflow_fix.value
    assert config["compression"]["quantize_outputs"] == params["advanced_parameters"].quantize_outputs

    assert config["compression"]["preset"] == params["preset"].value

    range_config = config["compression"]["initializer"]["range"]
    if isinstance(range_config, dict):
        assert range_config["num_init_samples"] == params["subset_size"]
        assert range_config["type"] == "mean_min_max"
    else:
        for rc in range_config:
            assert rc["num_init_samples"] == params["subset_size"]
            assert rc["type"] == "mean_min_max"

    num_bn_samples = config["compression"]["initializer"]["batchnorm_adaptation"]["num_bn_adaptation_samples"]
    if params["advanced_parameters"].disable_bias_correction is True:
        assert num_bn_samples == 0
    else:
        assert num_bn_samples == params["subset_size"]

    ref_scope = params["ignored_scope"].names if params["ignored_scope"] is not None else []
    assert config["compression"].get("ignored_scopes", []) == ref_scope

    # To validate NNCFConfig requared input_info
    config["input_info"] = {"sample_size": [1, 2, 224, 224]}
    NNCFConfig.validate(config)


@pytest.mark.parametrize("preset", (QuantizationPreset.MIXED, QuantizationPreset.PERFORMANCE))
@pytest.mark.parametrize("advanced_quantization_params", (AdvancedQuantizationParameters(),))
def test_apply_advanced_parameters_to_config(preset, advanced_quantization_params):
    compression_config = _get_default_quantization_config(preset, 1)
    assert apply_advanced_parameters_to_config(compression_config, advanced_quantization_params)
