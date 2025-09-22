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
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.model_graph_manager import get_const_data_by_name


def apply_magnitude_pruning(
    model: nn.Module,
    parameters: list[str],
    mode: PruneMode,
    ratio: float,
) -> nn.Module:
    """
    Prunes the specified parameters of the given model using unstructured pruning.

    This function registers hooks to the model's parameters to apply a binary mask
    for unstructured pruning based and update the specified ratio.

    Args:
        model (nn.Module): The neural network model to be pruned.
        parameters (list[str]): A list of parameter names to be pruned.
        mode (PruneMode): The mode of pruning to be applied.
        ratio (float): The ratio of parameters to prune.

    Returns:
        nn.Module: The pruned model with hooks registered for the specified parameters.
    """
    # Insert hooks
    pruned_param_names = set()
    for param_name in parameters:
        if param_name in pruned_param_names:
            # To avoid adding multiple hooks to a shared parameters
            continue
        pruned_param_names.add(param_name)
        param_data = get_const_data_by_name(param_name, model)
        hook_module = UnstructuredPruneBinaryMask(tuple(param_data.shape)).to(device=param_data.device)
        register_post_function_hook(
            model=model,
            op_name=param_name,
            port_id=0,
            hook=hook_module,
        )

    # Set ratio
    update_ratio(model, mode, ratio)

    return model


def get_pruned_modules(model: nn.Module) -> dict[str, UnstructuredPruneBinaryMask]:
    """
    Retrieves a mapping of operation names to their corresponding
    MagnitudeSparsityBinaryMask hooks from the given model.

    :param model: The model from which to retrieve the sparsity modules.
    :return: A dictionary mapping tensor names to their corresponding MagnitudeSparsityBinaryMask instances.
    """
    hook_storage = get_hook_storage(model)
    sparsity_module_map: dict[str, UnstructuredPruneBinaryMask] = dict()

    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, UnstructuredPruneBinaryMask):
            continue

        hook_type, op_name, port_id = decode_hook_name(name)
        if hook_type != "post_hooks" or port_id != 0:
            msg = f"Unexpected place of SparsityBinaryMask: {hook_type=}, {op_name=}, {port_id=}"
            raise nncf.InternalError(msg)
        sparsity_module_map[op_name] = hook

    return sparsity_module_map


@torch.no_grad()
def update_ratio(
    model: nn.Module,
    mode: PruneMode,
    ratio: float,
) -> None:
    """
    Updates the pruning ratio for the given model based on the specified pruning mode and sparsity level.

    This function modifies the binary masks of the sparsity modules in the model according to the
    specified pruning strategy. It calculates he threshold for pruning based on the absolute values of the weights
    and updates the binary masks accordingly.

    :param model: The neural network model to be pruned.
    :param mode: The mode of pruning to be applied.
    :param ratio: The desired pruning ratio, represented as a float between 0 and 1.
    """
    pruned_modules_map = get_pruned_modules(model)

    if not pruned_modules_map:
        msg = "No found Sparsity modules in the model"
        raise nncf.InternalError(msg)

    if mode == PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL:
        for const_name, hook in pruned_modules_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            threshold_index = int((data.numel() - 1) * ratio)
            abs_data = torch.abs(data)
            threshold = abs_data.view(-1).kthvalue(threshold_index + 1).values
            new_mask = (abs_data > threshold).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask = new_mask

    elif mode == PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL:
        # Get threshold value for all normalized weights
        all_weights: list[torch.Tensor] = []
        for const_name, hook in pruned_modules_map.items():
            data = get_const_data_by_name(const_name, model)
            all_weights.append((torch.abs(data) / data.norm(2)).view(-1))

        cat_all_weights = torch.cat(all_weights).view(-1)
        threshold_index = int((cat_all_weights.numel() - 1) * ratio)
        threshold_val = cat_all_weights.kthvalue(threshold_index + 1).values

        for const_name, hook in pruned_modules_map.items():
            data = get_const_data_by_name(const_name, model)

            # Calculate threshold and binary mask
            threshold_index = int((data.numel() - 1) * ratio)
            norm_data = torch.abs(data) / data.norm(2)
            new_mask = (norm_data > threshold_val).to(dtype=torch.bool)

            # Set new mask
            hook.binary_mask = new_mask
    else:
        msg = f"Unsupported pruning mode: {mode}"
        raise nncf.InternalError(msg)
