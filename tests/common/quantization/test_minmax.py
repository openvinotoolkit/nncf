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

import collections
import types

import pytest

import nncf
from nncf.common.quantization.structs import QuantizationPreset
from nncf.common.quantization.structs import QuantizationScheme
from nncf.common.quantization.structs import QuantizerGroup
from nncf.parameters import ModelType
from nncf.parameters import QuantizationMode
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import FP8QuantizationParameters
from nncf.quantization.advanced_parameters import FP8Type
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.quantization.advanced_parameters import QuantizationParameters
from nncf.quantization.algorithms.min_max.algorithm import MinMaxQuantization


@pytest.mark.parametrize(
    "preset,model_type,activation_mode,weights_mode",
    [
        (None, None, QuantizationScheme.SYMMETRIC, QuantizationScheme.SYMMETRIC),
        (QuantizationPreset.PERFORMANCE, None, QuantizationScheme.SYMMETRIC, QuantizationScheme.SYMMETRIC),
        (QuantizationPreset.MIXED, None, QuantizationScheme.ASYMMETRIC, QuantizationScheme.SYMMETRIC),
        (None, ModelType.TRANSFORMER, QuantizationScheme.ASYMMETRIC, QuantizationScheme.SYMMETRIC),
        (
            QuantizationPreset.PERFORMANCE,
            ModelType.TRANSFORMER,
            QuantizationScheme.SYMMETRIC,
            QuantizationScheme.SYMMETRIC,
        ),
        (QuantizationPreset.MIXED, ModelType.TRANSFORMER, QuantizationScheme.ASYMMETRIC, QuantizationScheme.SYMMETRIC),
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
    "algo_params, is_error",
    [
        ({"mode": None}, False),
        (
            {
                "mode": None,
                "activations_quantization_params": FP8QuantizationParameters(),
                "weights_quantization_params": FP8QuantizationParameters(),
            },
            True,
        ),
        ({"mode": QuantizationMode.FP8_E4M3}, False),
        (
            {
                "mode": QuantizationMode.FP8_E4M3,
                "preset": QuantizationPreset.MIXED,
            },
            True,
        ),
        (
            {"mode": QuantizationMode.FP8_E4M3, "target_device": TargetDevice.GPU},
            True,
        ),
        (
            {
                "mode": QuantizationMode.FP8_E4M3,
                "overflow_fix": OverflowFix.FIRST_LAYER,
            },
            True,
        ),
        (
            {
                "mode": QuantizationMode.FP8_E4M3,
                "quantize_outputs": True,
            },
            True,
        ),
        (
            {
                "mode": QuantizationMode.FP8_E4M3,
                "activations_quantization_params": QuantizationParameters(),
                "weights_quantization_params": QuantizationParameters(),
            },
            True,
        ),
        (
            {
                "mode": QuantizationMode.FP8_E4M3,
                "activations_quantization_params": QuantizationParameters(),
                "weights_quantization_params": QuantizationParameters(),
            },
            True,
        ),
    ],
)
def test_mode_against_default_map(algo_params, is_error):
    mode_param = algo_params["mode"]
    default_values_to_compare = {
        None: {
            "_overflow_fix": OverflowFix.FIRST_LAYER,
            "_activations_quantization_params": QuantizationParameters(),
            "_weights_quantization_params": QuantizationParameters(),
        },
        QuantizationMode.FP8_E4M3: {
            "_overflow_fix": OverflowFix.DISABLE,
            "_activations_quantization_params": FP8QuantizationParameters(FP8Type.E4M3),
            "_weights_quantization_params": FP8QuantizationParameters(FP8Type.E4M3),
        },
        QuantizationMode.FP8_E5M2: {
            "_overflow_fix": OverflowFix.DISABLE,
            "_activations_quantization_params": FP8QuantizationParameters(FP8Type.E5M2),
            "_weights_quantization_params": FP8QuantizationParameters(FP8Type.E5M2),
        },
    }

    qconf_attr_vs_constraint_dict_to_compare = {"mode": QuantizationScheme.SYMMETRIC}

    if is_error:
        with pytest.raises(nncf.ParameterNotSupportedError):
            minmax = MinMaxQuantization(**algo_params)
    else:
        minmax = MinMaxQuantization(**algo_params)
        for ref_parameter_name, ref_parameter_value in default_values_to_compare[mode_param].items():
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
            QuantizationMode.FP8_E4M3,
            None,
            None,
        ),
        (
            QuantizationMode.FP8_E5M2,
            None,
            None,
        ),
        (
            QuantizationMode.FP8_E4M3,
            FP8QuantizationParameters(destination_type=FP8Type.E4M3),
            FP8QuantizationParameters(destination_type=FP8Type.E4M3),
        ),
        (QuantizationMode.FP8_E4M3, FP8QuantizationParameters(destination_type=FP8Type.E5M2), None),
        (
            QuantizationMode.FP8_E5M2,
            None,
            FP8QuantizationParameters(destination_type=FP8Type.E4M3),
        ),
        (
            QuantizationMode.FP8_E5M2,
            FP8QuantizationParameters(destination_type=FP8Type.E4M3),
            FP8QuantizationParameters(destination_type=FP8Type.E4M3),
        ),
    ],
)
def test_mode_with_quantization_params(mode, activations_quantization_params, weights_quantization_params):
    minmax = MinMaxQuantization(
        mode=mode,
        activations_quantization_params=activations_quantization_params,
        weights_quantization_params=weights_quantization_params,
        overflow_fix=OverflowFix.DISABLE,
        preset=QuantizationPreset.PERFORMANCE,
    )
    default_configuration_map = {
        QuantizationMode.FP8_E4M3: FP8QuantizationParameters(destination_type=FP8Type.E4M3),
        QuantizationMode.FP8_E5M2: FP8QuantizationParameters(destination_type=FP8Type.E5M2),
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


def test_min_max_caching():
    """
    Checks that the _get_quantization_target_points(...) of MinMaxQuantization called once utilizing the cache.
    Checks that after _reset_cache() it called one more time.
    """
    find_called = 0
    fill_called = 0

    def find_qsetup_mock(self, *args):
        nonlocal find_called
        find_called += 1
        return

    def fill_qsetup_mock(self, *args):
        """
        Mocked _find_quantization_target_points.
        """
        nonlocal fill_called
        fill_called += 1
        # Set up cache
        self._quantization_target_points_to_qconfig = collections.OrderedDict()
        self._unified_scale_groups = []
        return self._quantization_target_points_to_qconfig, self._unified_scale_groups

    run_nums = 2
    algo = MinMaxQuantization()
    algo.find_quantization_setup = types.MethodType(find_qsetup_mock, algo)
    algo.fill_quantization_target_points = types.MethodType(fill_qsetup_mock, algo)
    for _ in range(run_nums):
        algo._get_quantization_target_points(None, None)
    assert find_called == fill_called == 1
    algo._reset_cache()
    for _ in range(run_nums):
        algo._get_quantization_target_points(None, None)
    assert find_called == fill_called == 2


@pytest.mark.parametrize(
    "num_bits, target_device, ref_hw_target_device",
    [
        (8, target_device, target_device)
        for target_device in TargetDevice
        if target_device not in [TargetDevice.GPU, TargetDevice.NPU]
    ]
    + [
        (8, TargetDevice.NPU, TargetDevice.CPU),
        (8, TargetDevice.GPU, TargetDevice.CPU),
        (4, TargetDevice.NPU, TargetDevice.NPU),
    ],
)
def test_target_device(num_bits, target_device, ref_hw_target_device):
    min_max_algo = MinMaxQuantization(
        target_device=target_device,
        activations_quantization_params=QuantizationParameters(num_bits=num_bits),
        weights_quantization_params=QuantizationParameters(num_bits=num_bits),
    )
    assert min_max_algo._target_device == ref_hw_target_device


@pytest.mark.parametrize("activation_bits", [8, 4])
@pytest.mark.parametrize("weight_bits", [8, 4])
def test_overflow_fix(activation_bits, weight_bits):
    quant_scheme_a8w8 = activation_bits == 8 and weight_bits == 8

    min_max_algo = MinMaxQuantization(
        activations_quantization_params=QuantizationParameters(num_bits=activation_bits),
        weights_quantization_params=QuantizationParameters(num_bits=weight_bits),
    )

    if quant_scheme_a8w8:
        assert min_max_algo._overflow_fix == OverflowFix.FIRST_LAYER
    else:
        assert min_max_algo._overflow_fix == OverflowFix.DISABLE
