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
import sys
from importlib import import_module
from pathlib import Path

import pytest
import torch
from torch import nn

import nncf
from nncf.common.quantization.structs import QuantizationScheme
from nncf.torch import get_config
from nncf.torch import load_from_config
from nncf.torch.function_hook import get_hook_storage
from nncf.torch.function_hook import register_post_function_hook
from nncf.torch.function_hook import register_pre_function_hook
from nncf.torch.function_hook import wrap_model
from nncf.torch.function_hook.serialization import MODULE_NAME_MAP
from nncf.torch.layer_utils import StatefulModuleInterface
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
                "module_path": "tests.torch.function_hook.helpers",
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

    with pytest.raises(nncf.InternalError, match="StatefulModuleInterface"):
        get_config(model)


@pytest.mark.parametrize(
    "module_cls",
    (
        "UnstructuredPruningMask",
        "RBPruningMask",
        "SymmetricQuantizer",
        "AsymmetricQuantizer",
        "AsymmetricLoraQuantizer",
        "AsymmetricLoraNLSQuantizer",
        "SymmetricLoraQuantizer",
        "SymmetricLoraNLSQuantizer",
        "SQMultiply",
    ),
)
def test_backward_compatibility_map(module_cls: str):
    # validate that all module classes from MODULE_NAME_MAP are importable and are subclasses of StatefulModuleInterface
    imported_module = import_module(MODULE_NAME_MAP[module_cls])
    module_cls = getattr(imported_module, module_cls)
    assert issubclass(module_cls, StatefulModuleInterface)


@pytest.fixture()
def _restore_sys_modules():
    original_modules = sys.modules.copy()
    yield
    sys.modules = original_modules


@pytest.mark.parametrize("with_module_path", [True, False], ids=["new_config", "legacy_config"])
def test_load_from_config_without_manual_compression_class_import(with_module_path: bool, _restore_sys_modules: None):
    module_path = "nncf.torch.quantization.layers"
    sys.modules.pop(module_path, None)

    compression_state = {
        "hook_names_in_model": ["pre_hooks.conv1/conv2d/0__0.0"],
        "module_cls_name": "AsymmetricQuantizer",
        "module_config": {
            "num_bits": 8,
            "mode": QuantizationScheme.ASYMMETRIC,
            "signedness_to_force": True,
            "narrow_range": False,
            "half_range": False,
            "scale_shape": (1,),
            "logarithm_scale": False,
            "is_quantized_on_export": False,
            "compression_lr_multiplier": None,
        },
    }
    if with_module_path:
        compression_state["module_path"] = "nncf.torch.quantization.layers"

    restored_model = load_from_config(SimpleModel(), {"compression_state": [compression_state]})

    hook_storage = get_hook_storage(restored_model)
    quantizer = hook_storage.get_submodule("pre_hooks.conv1/conv2d/0__0.0")

    assert quantizer.__class__.__name__ == "AsymmetricQuantizer"
    assert quantizer.__class__.__module__ == module_path

    assert module_path in sys.modules
