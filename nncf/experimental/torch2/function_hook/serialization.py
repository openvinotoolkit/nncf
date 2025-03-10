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

from typing import Any, Dict, List, TypedDict, TypeVar, cast
from weakref import WeakKeyDictionary

from torch import nn

import nncf
from nncf.experimental.torch2.function_hook.wrapper import get_hook_storage
from nncf.experimental.torch2.function_hook.wrapper import wrap_model
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import StatefulModuleInterface

COMPRESSION_STATE_ATTR = "compression_state"
TModel = TypeVar("TModel", bound=nn.Module)


class S_COMMAND(TypedDict):
    hook_names_in_model: List[str]
    module_cls_name: str
    module_config: Dict[str, Any]


def get_config(model: nn.Module) -> Dict[str, Any]:
    """
    Returns serializable config which contains all information required to recover all additional modules placement.

    :param model: The model to serialize.
    :return: Serializable config.
    """
    hook_storage = get_hook_storage(model)

    # Find shared modules
    modules_map: WeakKeyDictionary[nn.Module, List[str]] = WeakKeyDictionary()
    for name, module in hook_storage.named_modules(remove_duplicate=False):
        splitted_name = name.split(".")
        if len(splitted_name) != 3:
            # Expected depths of target hook module is 3
            # <3 - ModuleDicts in HookStorage, >3 - submodules of hooks
            continue
        if module not in modules_map:
            modules_map[module] = []
        modules_map[module].append(name)

    # Generate serialized transformation commands
    serialized_transformations: List[S_COMMAND] = []
    for module, names in modules_map.items():
        compression_module_name = module.__class__.__name__
        if compression_module_name not in COMPRESSION_MODULES.registry_dict:
            msg = (
                f"Could not serialize compression module with name {compression_module_name}. "
                "Please register your module in the COMPRESSION_MODULES registry."
            )
            raise nncf.InternalError(msg)
        if not isinstance(module, StatefulModuleInterface):
            msg = "Support only StatefulModuleInterface modules"
            raise nncf.InternalError(msg)

        serialized_transformations.append(
            {
                "hook_names_in_model": names,
                "module_cls_name": compression_module_name,
                "module_config": module.get_config(),
            }
        )

    return {COMPRESSION_STATE_ATTR: serialized_transformations}


def load_from_config(model: TModel, config: Dict[str, Any]) -> TModel:
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
    transformation_commands = cast(List[S_COMMAND], config[COMPRESSION_STATE_ATTR])
    for command in transformation_commands:
        module_cls = COMPRESSION_MODULES.get(command["module_cls_name"])
        module = module_cls.from_config(command["module_config"])
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
