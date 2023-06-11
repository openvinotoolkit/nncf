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

from nncf.common.quantization.structs import QuantizationPreset
from nncf.parameters import ModelType
from nncf.parameters import TargetDevice
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.scopes import IgnoredScope
from nncf.tensorflow.quantization.quantize_model import _create_nncf_config


@pytest.mark.parametrize(
    "params",
    (
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 1,
            "model_type": ModelType.TRANSFORMER,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE, quantize_outputs=True, disable_bias_correction=True
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 2,
            "model_type": None,
            "ignored_scope": None,
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.ENABLE, quantize_outputs=False, disable_bias_correction=False
            ),
        },
        {
            "preset": QuantizationPreset.MIXED,
            "target_device": TargetDevice.ANY,
            "subset_size": 3,
            "model_type": None,
            "ignored_scope": IgnoredScope(names=["node_1"]),
            "advanced_parameters": AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.FIRST_LAYER, quantize_outputs=True, disable_bias_correction=False
            ),
        },
    ),
)
def test_create_nncf_config(params):
    config = _create_nncf_config(**params)

    assert config["compression"]["overflow_fix"] == params["advanced_parameters"].overflow_fix.value
    assert config["compression"]["quantize_outputs"] == params["advanced_parameters"].quantize_outputs

    assert config["compression"]["preset"] == params["preset"].value
    assert config["compression"]["initializer"]["range"]["num_init_samples"] == params["subset_size"]

    num_bn_samples = config["compression"]["initializer"]["batchnorm_adaptation"]["num_bn_adaptation_samples"]
    if params["advanced_parameters"].disable_bias_correction is True or params["model_type"] == ModelType.TRANSFORMER:
        assert num_bn_samples == 0
    else:
        assert num_bn_samples == params["subset_size"]

    ref_scope = params["ignored_scope"].names if params["ignored_scope"] is not None else []
    if params["model_type"] == ModelType.TRANSFORMER:
        ref_scope = [
            "{re}.*Embeddings.*",
            "{re}.*__add___[0-1]",
            "{re}.*layer_norm_0",
            "{re}.*matmul_1",
            "{re}.*__truediv__*",
        ] + ref_scope
    assert config["compression"].get("ignored_scopes", []) == ref_scope
