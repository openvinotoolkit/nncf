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

from copy import deepcopy
from typing import Optional

from torch import nn

try:
    from nncf.experimental.torch.replace_custom_modules.timm_custom_modules import replace_timm_modules
except ImportError:
    replace_timm_modules = None


def try_convert_module(module: nn.Module) -> Optional[nn.Module]:
    """
    Tries to convert custom module to PyTorch native modules if possible using the replace_timm_modules function.

    :param module: The module to try to convert.
    :return nn.Module: The replaced module if the function succeeded in replacing the module, None otherwise.
    """
    replaced_module = None
    if replace_timm_modules is not None:
        replaced_module = replace_timm_modules(module)
    return replaced_module


def replace_modules(target_module: nn.Module):
    """
    Recursively walks through all modules in the target module and replaces custom modules with PyTorch native modules.

    :param target_module: The target module to replace modules in.
    """
    if type(target_module) == nn.Sequential:
        for idx in range(len(target_module)):
            if type(target_module[idx]) == nn.Sequential:
                replace_modules(target_module[idx])
            else:
                replaced_module = try_convert_module(target_module[idx])
                if replaced_module is not None:
                    target_module[idx] = replaced_module
                else:
                    replace_modules(target_module[idx])
    else:
        for name_child_module, child_module in target_module.named_children():
            if type(child_module) == nn.Sequential:
                replace_modules(child_module)
            else:
                replaced_module = try_convert_module(child_module)
                if replaced_module is not None:
                    setattr(target_module, name_child_module, replaced_module)
                else:
                    replace_modules(child_module)


def replace_custom_modules_with_torch_native(model: nn.Module) -> nn.Module:
    """
    Replace custom module that can not be operated by NNCF to torch native modules.

    :param model: The target model.
    :return nn.Module: Transformed model.
    """
    new_model = deepcopy(model)

    replaced_module = try_convert_module(model)
    if replaced_module is not None:
        return replaced_module

    replace_modules(new_model)
    return new_model
