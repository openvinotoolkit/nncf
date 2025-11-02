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
from typing import TypeVar

import torch
from torch import nn

import nncf
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.pruning.magnitude.modules import UnstructuredPruningMask
from nncf.torch.function_hook.pruning.rb.modules import RBPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_graph_manager import get_module_by_name
from nncf.torch.model_graph_manager import split_const_name

TModel = TypeVar("TModel", bound=nn.Module)


@torch.no_grad()
def apply_pruning_in_place(model: TModel) -> TModel:
    """
    Applies pruning masks in-place to the weights:
        (weights + pruning mask) -> (pruned weights)

    :param model: Compressed model
    :return: The modified NNCF network.
    """
    hook_storage = get_hook_storage(model)
    hooks_to_delete = []
    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, (RBPruningMask, UnstructuredPruningMask)):
            continue
        hook.eval()
        hook_type, op_name, port_id = decode_hook_name(name)
        if hook_type != "post_hooks" or port_id != 0:
            msg = f"Unexpected place of SparsityBinaryMask: {hook_type=}, {op_name=}, {port_id=}"
            raise nncf.InternalError(msg)

        module_name, weight_attr_name = split_const_name(op_name)
        module = get_module_by_name(module_name, model)
        weight_param = getattr(module, weight_attr_name)

        weight_param.requires_grad = False
        weight_param.data = hook(weight_param)

        hooks_to_delete.append(name)

    for hook_name in hooks_to_delete:
        hook_storage.delete_hook(hook_name)

    return model
