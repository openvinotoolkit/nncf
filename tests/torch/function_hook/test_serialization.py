# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pytest
import torch
from torch import nn

import nncf
from nncf.torch import get_config
from nncf.torch import load_from_config
from nncf.torch.function_hook import get_hook_storage
from nncf.torch.function_hook import register_post_function_hook
from nncf.torch.function_hook import register_pre_function_hook
from nncf.torch.function_hook import wrap_model
from tests.torch.function_hook.helpers import HookWithState
from tests.torch.function_hook.helpers import SimpleModel


@pytest.mark.parametrize("is_shared_hook", [True, False], ids=["shared_hook", "not_shared_hook"])
def test_save_load(tmp_path: Path, is_shared_hook: bool, use_cuda: bool):
    device = "cuda" if use_cuda else "cpu"
    model = wrap_model(SimpleModel().to(device))

    hook1 = HookWithState("hook1")
    hook2 = hook1 if is_shared_hook else HookWithState("hook2")

    register_pre_function_hook(model, "conv1/conv2d/0", 0, hook1)
    register_post_function_hook(model, "simple/conv/conv2d/0", 0, hook2)

    state_dict = model.state_dict()
    compression_config = get_config(model)

    torch.save(
        {
            "model_state_dict": state_dict,
            "compression_config": compression_config,
        },
        tmp_path / "checkpoint.pth",
    )

    ckpt = torch.load(tmp_path / "checkpoint.pth")
    config = ckpt["compression_config"]
    restored_model = load_from_config(SimpleModel().to(device), config)
    restored_model.load_state_dict(ckpt["model_state_dict"])

    assert state_dict == restored_model.state_dict()

    tensor = model.get_example_inputs().to(device)
    ret_1 = model(tensor)
    ret_2 = restored_model(tensor)
    assert torch.allclose(ret_1[0], ret_2[0])
    assert torch.allclose(ret_1[1], ret_2[1])

    hook_storage = get_hook_storage(restored_model)
    hook1 = hook_storage.get_submodule("pre_hooks.conv1/conv2d/0__0.0")
    hook2 = hook_storage.get_submodule("post_hooks.simple/conv/conv2d/0__0.0")
    assert (hook1 is hook2) == is_shared_hook


def test_error_duplicate_names():
    config = {
        "compression_state": [
            {
                "hook_names_in_model": ["pre_hooks.conv1/conv2d/0__0.0", "pre_hooks.conv1/conv2d/0__0.0"],
                "module_cls_name": "HookWithState",
                "module_config": "hook1",
            }
        ]
    }
    with pytest.raises(nncf.InternalError, match="already registered"):
        load_from_config(SimpleModel(), config)


def test_error_not_registered_compression_modules():
    model = wrap_model(SimpleModel())
    register_pre_function_hook(model, "conv1/conv2d/0", 0, nn.ReLU())

    with pytest.raises(nncf.InternalError, match="Please register your module in the COMPRESSION_MODULES registry."):
        get_config(model)
