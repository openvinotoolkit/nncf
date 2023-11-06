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
