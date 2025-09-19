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
import torch
from torch import nn

import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.prune.magnitude.modules import UnstructuredPruneBinaryMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.model_graph_manager import get_const_data_by_name


def get_sparsity_modules(model: nn.Module) -> dict[str, UnstructuredPruneBinaryMask]:
    """
    Retrieves a mapping of operation names to their corresponding
    MagnitudeSparsityBinaryMask hooks from the given model.

    :param model: The model from which to retrieve the sparsity modules.
    :return: A dictionary mapping tensor names to their corresponding MagnitudeSparsityBinaryMask instances.
    """
    hook_storage = get_hook_storage(model)
    sparsity_module_map: dict[str, UnstructuredPruneBinaryMask] = dict()

    for name, hook in hook_storage.named_hooks():
        if isinstance(hook, UnstructuredPruneBinaryMask):
            continue

        hook_type, op_name, port_id = decode_hook_name(name)
        if hook_type != "post_hooks" or port_id != 0:
            msg = f"Unexpected place of SparsityBinaryMask: {hook_type=}, {op_name=}, {port_id=}"
            raise nncf.InternalError(msg)
        sparsity_module_map[op_name] = hook

    return sparsity_module_map


def prune_update_ratio(
    model: nn.Module,
    mode: PruneMode,
    sparsity_level: float,
) -> None:
    """
    Updates the pruning ratio for the given model based on the specified pruning mode and sparsity level.

    This function modifies the binary masks of the sparsity modules in the model according to the
    specified pruning strategy. It calculates he threshold for pruning based on the absolute values of the weights
    and updates the binary masks accordingly.

    :param model: The neural network model to be pruned.
    :param mode: The mode of pruning to be applied.
    :param sparsity_level: The desired level of sparsity, represented as a float between 0 and 1.
    """
    sparsity_module_map = get_sparsity_modules(model)

    if not sparsity_module_map:
        msg = "No found Sparsity modules in the model"
        return nncf.InternalError(msg)

    if mode == PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL:
        for const_name, hook in sparsity_module_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            with torch.no_grad():
                threshold_index = int((data.numel() - 1) * sparsity_level)
                abs_data = torch.abs(data)
                threshold = abs_data.view(-1).kthvalue(threshold_index).values
                new_mask = (abs_data >= threshold).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask = new_mask

    elif mode == PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL:
        # Get threshold value for all normalized weights
        all_weights: list[torch.Tensor] = []
        for const_name, hook in sparsity_module_map.items():
            data = get_const_data_by_name(const_name, model)

            with torch.no_grad():
                all_weights.append((torch.abs(data) / data.norm(2)).view(-1))

        cat_all_weights = torch.cat(all_weights).view(-1)
        threshold_index = int((cat_all_weights.numel() - 1) * sparsity_level)
        threshold_val = cat_all_weights.kthvalue(threshold_index).values

        for const_name, hook in sparsity_module_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            with torch.no_grad():
                threshold_index = int((data.numel() - 1) * sparsity_level)
                norm_data = torch.abs(data) / data.norm(2)
                threshold = norm_data.view(-1).kthvalue(threshold_index).values
                new_mask = (norm_data >= threshold_val).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask = new_mask
    else:
        msg = f"Unsupported pruning mode: {mode}"
        raise nncf.InternalError(msg)
