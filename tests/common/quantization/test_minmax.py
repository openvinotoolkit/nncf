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

import pytest

from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizerGroup
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode as Mode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import ConvertParameters
from nncf.quantization.advanced_parameters import FP8Type
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization


@pytest.mark.parametrize(
    "preset,model_type,activation_mode,weights_mode",
    [
        (None, None, QuantizationMode.SYMMETRIC, QuantizationMode.SYMMETRIC),
        (QuantizationPreset.PERFORMANCE, None, QuantizationMode.SYMMETRIC, QuantizationMode.SYMMETRIC),
        (QuantizationPreset.MIXED, None, QuantizationMode.ASYMMETRIC, QuantizationMode.SYMMETRIC),
        (None, ModelType.TRANSFORMER, QuantizationMode.ASYMMETRIC, QuantizationMode.SYMMETRIC),
        (QuantizationPreset.PERFORMANCE, ModelType.TRANSFORMER, QuantizationMode.SYMMETRIC, QuantizationMode.SYMMETRIC),
        (QuantizationPreset.MIXED, ModelType.TRANSFORMER, QuantizationMode.ASYMMETRIC, QuantizationMode.SYMMETRIC),
    ],
)
def test_quantization_preset(preset, model_type, activation_mode, weights_mode):
    minmax = MinMaxQuantization(preset=preset, model_type=model_type)

    global_quantizer_constraints = getattr(minmax, "_global_quantizer_constraints")
    assert (
        global_quantizer_constraints[QuantizerGroup.ACTIVATIONS].qconf_attr_vs_constraint_dict["mode"]
        == activation_mode
    )
    assert global_quantizer_constraints[QuantizerGroup.WEIGHTS].qconf_attr_vs_constraint_dict["mode"] == weights_mode


@pytest.mark.parametrize(
    "algo_params",
    [
        {"mode": Mode.FP8_E4M3},
        {
            "mode": Mode.FP8_E4M3,
            "preset": QuantizationPreset.PERFORMANCE,
            "target_device": TargetDevice.CPU,
            "overflow_fix": OverflowFix.DISABLE,
            "quantize_outputs": False,
            "backend_params": None,
        },
        {
            "mode": Mode.FP8_E4M3,
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.GPU,
            "overflow_fix": OverflowFix.FIRST_LAYER,
            "quantize_outputs": True,
        },
        {
            "mode": Mode.FP8_E4M3,
            "target_device": TargetDevice.CPU_SPR,
            "overflow_fix": OverflowFix.ENABLE,
        },
    ],
)
def test_mode_against_default_map(algo_params):
    default_values_to_compare = {
        "_preset": QuantizationPreset.PERFORMANCE,
        "_target_device": TargetDevice.CPU,
        "_overflow_fix": OverflowFix.DISABLE,
        "_quantize_outputs": False,
        "_backend_params": None,
    }

    qconf_attr_vs_constraint_dict_to_compare = {"mode": QuantizationMode.SYMMETRIC}

    minmax = MinMaxQuantization(**algo_params)
    for ref_parameter_name, ref_parameter_value in default_values_to_compare.items():
        parameter_value = getattr(minmax, ref_parameter_name)
        assert parameter_value == ref_parameter_value

        global_quantizer_constraints = getattr(minmax, "_global_quantizer_constraints")
        assert (
            global_quantizer_constraints[QuantizerGroup.ACTIVATIONS].qconf_attr_vs_constraint_dict
            == qconf_attr_vs_constraint_dict_to_compare
        )
        assert (
            global_quantizer_constraints[QuantizerGroup.WEIGHTS].qconf_attr_vs_constraint_dict
            == qconf_attr_vs_constraint_dict_to_compare
        )


@pytest.mark.parametrize(
    "mode, activations_quantization_params, weights_quantization_params",
    [
        (
            Mode.FP8_E4M3,
            None,
            None,
        ),
        (
            Mode.FP8_E5M2,
            None,
            None,
        ),
        (
            Mode.FP8_E4M3,
            ConvertParameters(destination_type=FP8Type.E4M3),
            ConvertParameters(destination_type=FP8Type.E4M3),
        ),
        (Mode.FP8_E4M3, ConvertParameters(destination_type=FP8Type.E5M2), None),
        (
            Mode.FP8_E5M2,
            None,
            ConvertParameters(destination_type=FP8Type.E4M3),
        ),
        (
            Mode.FP8_E5M2,
            ConvertParameters(destination_type=FP8Type.E4M3),
            ConvertParameters(destination_type=FP8Type.E4M3),
        ),
    ],
)
def test_mode_with_quantization_params(mode, activations_quantization_params, weights_quantization_params):
    minmax = MinMaxQuantization(
        mode=mode,
        activations_quantization_params=activations_quantization_params,
        weights_quantization_params=weights_quantization_params,
    )
    default_configuration_map = {
        Mode.FP8_E4M3: ConvertParameters(destination_type=FP8Type.E4M3),
        Mode.FP8_E5M2: ConvertParameters(destination_type=FP8Type.E5M2),
    }

    quantization_params = getattr(minmax, "_quantization_params")
    assert (
        quantization_params[QuantizerGroup.ACTIVATIONS] == default_configuration_map[mode]
        if activations_quantization_params is None
        else activations_quantization_params
    )
    assert (
        quantization_params[QuantizerGroup.WEIGHTS] == default_configuration_map[mode]
        if weights_quantization_params is None
        else weights_quantization_params
    )
