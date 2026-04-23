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

from importlib import import_module
from typing import Any, TypedDict, TypeVar, cast

from torch import nn

import nncf
from nncf.common.logging import nncf_logger
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.layer_utils import StatefulModuleInterface
from nncf.torch.utils import get_model_device
from nncf.torch.utils import is_multidevice

COMPRESSION_STATE_ATTR = "compression_state"
TModel = TypeVar("TModel", bound=nn.Module)


class S_COMMAND(TypedDict):
    hook_names_in_model: list[str]
    module_path: str
    module_cls_name: str
    module_config: dict[str, Any]


def get_config(model: nn.Module) -> dict[str, Any]:
    """
    Returns serializable config which contains all information required to recover all additional modules placement.

    :param model: The model to serialize.
    :return: Serializable config.
    """
    hook_storage = get_hook_storage(model)

    # Find shared modules
    modules_map: dict[nn.Module, list[str]] = dict()
    for name, module in hook_storage.named_hooks(remove_duplicate=False):
        if module not in modules_map:
            modules_map[module] = []
        modules_map[module].append(name)

    # Generate serialized transformation commands
    serialized_transformations: list[S_COMMAND] = []
    for module, names in modules_map.items():
        compression_module_name = module.__class__.__name__
        if not isinstance(module, StatefulModuleInterface):
            msg = "Support only StatefulModuleInterface modules"
            raise nncf.InternalError(msg)

        serialized_transformations.append(
            {
                "hook_names_in_model": names,
                "module_path": module.__class__.__module__,
                "module_cls_name": compression_module_name,
                "module_config": module.get_config(),
            }
        )

    return {COMPRESSION_STATE_ATTR: serialized_transformations}


# Use to map module class name to module import path for backward compatibility
# TODO(AlexanderDokuchaev): Remove after several release (expected: 3.4.0)
MODULE_NAME_MAP = {
    "UnstructuredPruningMask": "nncf.torch.function_hook.pruning.magnitude.modules",
    "RBPruningMask": "nncf.torch.function_hook.pruning.rb.modules",
    "SymmetricQuantizer": "nncf.torch.quantization.layers",
    "AsymmetricQuantizer": "nncf.torch.quantization.layers",
    "AsymmetricLoraQuantizer": "nncf.torch.quantization.layers",
    "AsymmetricLoraNLSQuantizer": "nncf.torch.quantization.layers",
    "SymmetricLoraQuantizer": "nncf.torch.quantization.layers",
    "SymmetricLoraNLSQuantizer": "nncf.torch.quantization.layers",
    "SQMultiply": "nncf.torch.quantization.layers",
}


def load_from_config(model: TModel, config: dict[str, Any]) -> TModel:
    """
    Initialize model with compressed modules from config file.

    .. code-block:: python

        model = MyModel()
        qmodel = nncf.quantize(model, ...)
        torch.save(
            {
                "state_dict": qmodel.state_dict(),
                "config": get_config(qmodel),
            },
            "ckpt.pth",
        )
        ...
        ckpt = torch.load("ckpt.pth")
        restored_model = load_from_config(MyModel(), ckpt["config"])
        restored_model.load_state_dict(ckpt["state_dict"])

    :param model: The original uncompressed model.
    :param config: The configuration dictionary containing the compressed model information.
    :return: The compressed model.
    """
    wrapped_model = wrap_model(model)
    hook_storage = get_hook_storage(wrapped_model)
    transformation_commands = cast(list[S_COMMAND], config[COMPRESSION_STATE_ATTR])

    device = None
    if not is_multidevice(wrapped_model):
        device = get_model_device(wrapped_model)
    else:
        nncf_logger.warning("Model is on multiple devices. Cannot determine device for loaded modules.")

    for command in transformation_commands:
        module_path = command.get("module_path") or MODULE_NAME_MAP[command["module_cls_name"]]
        imported_module = import_module(module_path)
        module_cls = getattr(imported_module, command["module_cls_name"])
        if not issubclass(module_cls, StatefulModuleInterface):
            msg = "Support only StatefulModuleInterface modules"
            raise nncf.InternalError(msg)
        module = module_cls.from_config(command["module_config"])
        module.to(device)
        for target_name in command["hook_names_in_model"]:
            hook_type, hook_key, hook_id = target_name.split(".")
            storage_dict = getattr(hook_storage, hook_type)
            if hook_key not in storage_dict:
                storage_dict[hook_key] = nn.ModuleDict()
            if hook_id in storage_dict[hook_key]:
                msg = f"{hook_id=} for {hook_type}.{hook_key} already registered"
                raise nncf.InternalError(msg)
            storage_dict[hook_key][hook_id] = module
    return wrapped_model
