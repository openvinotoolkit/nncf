# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Tuple

import torch
from torch import nn

import nncf


def split_const_name(const_name: str) -> Tuple[str, str]:
    """
    Splits the constant name into module and attribute names.

    :param const_name: The full name of the constant, including module and attribute.
    :return:
        - module_name: The name of the module containing the constant.
        - weight_attr_name: The name of the constant attribute within the module.
    """
    index = const_name.rfind(".")
    if index == -1:
        return str(), const_name
    module_name = const_name[:index]
    weight_attr_name = const_name[index + 1 :]
    return module_name, weight_attr_name


def get_module_by_name(module_name: str, model: torch.nn.Module) -> torch.nn.Module:
    """
    Retrieves a module from a PyTorch model by its hierarchical name.

    :param module_name: The name of the module to retrieve (e.g., "module1.submodule2").
    :param model: The PyTorch model.
    :return: The retrieved module.
    """
    if not module_name:
        return model
    curr_module = model
    for name in module_name.split("."):
        for child_name, child_module in curr_module.named_children():
            if child_name == name:
                curr_module = child_module
                break
        else:
            raise nncf.ModuleNotFoundError(f"Could not find the {module_name} module in the model.")
    return curr_module


def get_const_data(model: nn.Module, const_name: str) -> torch.Tensor:
    """
    Retrieves a constant tensor associated with a given node.

    :param const_name: The name of const data.
    :param model: The PyTorch model.
    :return: A torch.Tensor object containing the constant value.
    """
    module_name, const_attr_name = split_const_name(const_name)
    module = get_module_by_name(module_name, model)
    data: torch.Tensor = getattr(module, const_attr_name)
    if isinstance(data, torch.nn.Parameter):
        return data.data
    return data


def set_const_data(model: nn.Module, const_name: str, data: torch.Tensor) -> None:
    """
    Sets the constant data associated with a specific name of a tensor in a PyTorch model.

    :param model: The PyTorch model.
    :param const_name: The name of tensor in the model.
    :param data: The constant data tensor to be set.
    """
    module_name, const_attr_name = split_const_name(const_name)
    module = get_module_by_name(module_name, model)
    const = getattr(module, const_attr_name)
    if isinstance(const, torch.nn.Parameter):
        const.data = data
    else:
        setattr(module, const_attr_name, data)
