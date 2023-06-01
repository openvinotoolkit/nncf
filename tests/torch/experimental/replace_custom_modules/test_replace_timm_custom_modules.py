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
import timm
import torch
from timm.layers import Linear
from timm.layers.norm_act import BatchNormAct2d
from timm.layers.norm_act import GroupNormAct
from timm.layers.norm_act import LayerNormAct
from torch import nn

from nncf.experimental.torch.replace_custom_modules.timm_custom_modules import convert_timm_custom_modules
from nncf.experimental.torch.replace_custom_modules.timm_custom_modules import (
    replace_timm_custom_modules_with_torch_native,
)


def _count_custom_modules(model) -> int:
    """
    Get number of custom modules in the model.
    :param model: The target model.
    :return int: Number of custom modules.
    """
    custom_types = [
        Linear,
        BatchNormAct2d,
        GroupNormAct,
        LayerNormAct,
    ]
    return len([m for _, m in model.named_modules() if type(m) in custom_types])


TEST_CUSTOM_MODULES = [
    Linear(
        in_features=2,
        out_features=2,
    ),
    BatchNormAct2d(
        num_features=2,
        act_layer=nn.ReLU,
    ),
    GroupNormAct(
        num_channels=2,
        num_groups=2,
        act_layer=nn.ReLU,
    ),
    LayerNormAct(
        normalization_shape=(2, 2),
        act_layer=nn.ReLU,
    ),
]


@pytest.mark.parametrize("custom_module", TEST_CUSTOM_MODULES, ids=[m.__class__.__name__ for m in TEST_CUSTOM_MODULES])
@pytest.mark.skipif(timm is None, reason="Not install timm package")
def test_replace_custom_timm_module(custom_module):
    """
    Test output of replaced module with custom module
    """
    native_module = convert_timm_custom_modules(custom_module)
    input_data = torch.rand(1, 2, 2, 2)
    out_custom = custom_module(input_data)
    out_native = native_module(input_data)

    assert type(custom_module) != type(native_module)
    assert torch.equal(out_custom, out_native)


def test_replace_custom_modules_in_timm_model():
    """
    Test that all modules from timm model replaced by replace_custom_modules_with_torch_native
    """
    timm_model = timm.create_model(
        "mobilenetv3_small_050", num_classes=1000, in_chans=3, pretrained=True, checkpoint_path=""
    )
    input_data = torch.rand(1, 3, 224, 224)
    out_timm = timm_model(input_data)

    native_model = replace_timm_custom_modules_with_torch_native(timm_model)
    out_native = native_model(input_data)
    assert torch.equal(out_timm, out_native)

    num_custom_modules_in_timm = _count_custom_modules(timm_model)
    num_custom_modules_in_native = _count_custom_modules(native_model)

    assert num_custom_modules_in_native == 0
    assert num_custom_modules_in_timm > 0
