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


import torch
from torch import nn

from nncf.pruning.prune_model import ModelPruningStatistic
from nncf.pruning.prune_model import TensorPruningStatistic
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.pruning.magnitude.modules import UnstructuredPruningMask
from nncf.torch.function_hook.pruning.rb.modules import RBPruningMask
from nncf.torch.function_hook.pruning.rb.modules import binary_mask
from nncf.torch.function_hook.wrapper import get_hook_storage


@torch.no_grad()
def pruning_statistic(model: nn.Module) -> ModelPruningStatistic:
    """
    Collects and returns pruning statistics for the given model.

    :param model: The pruned model.
    :return: Pruning statistics.
    """
    total_params = sum(p.numel() for p in model.parameters())
    num_elements = 0
    pruned_elements = 0
    stat_per_tensors: list[TensorPruningStatistic] = []

    hook_storage = get_hook_storage(model)
    for hook_name, hook_module in hook_storage.named_hooks():
        if isinstance(hook_module, UnstructuredPruningMask):
            mask = hook_module.binary_mask
        elif isinstance(hook_module, RBPruningMask):
            mask = binary_mask(hook_module.mask)
            # Exclude RBPruningMask’s internal mask parameters from the total parameter count
            total_params -= mask.numel()
        else:
            continue

        pruned_el = int(torch.sum(mask == 0).item())
        num_el = mask.numel()
        shape = tuple(mask.shape)
        pruned_ratio = pruned_el / num_el if num_el != 0 else 0.0

        _, tensor_name, _ = decode_hook_name(hook_name)

        num_elements += num_el
        pruned_elements += pruned_el

        stat_per_tensors.append(TensorPruningStatistic(tensor_name, shape, pruned_ratio))

    masked_ratio = pruned_elements / num_elements if num_elements != 0 else 0.0
    global_ratio = pruned_elements / total_params if total_params != 0 else 0.0

    return ModelPruningStatistic(
        pruning_ratio=masked_ratio,
        global_pruning_ratio=global_ratio,
        pruned_tensors=stat_per_tensors,
    )
