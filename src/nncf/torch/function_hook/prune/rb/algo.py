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

from torch import nn

import nncf
from nncf.torch.function_hook.hook_storage import decode_hook_name
from nncf.torch.function_hook.prune.rb.modules import RBPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.model_graph_manager import get_const_data_by_name


def apply_regularization_based_pruning(
    model: nn.Module,
    parameters: list[str],
) -> nn.Module:
    """
    :param model: The neural network model to be pruned.
    :param parameters: A list of parameter names to be pruned.
    :param ratio: The ratio of parameters to prune.
    :returns: The pruned model with hooks registered for the specified parameters.
    """
    # Insert hooks
    for param_name in set(parameters):
        param_data = get_const_data_by_name(param_name, model)
        hook_module = RBPruningMask(shape=tuple(param_data.shape)).to(device=param_data.device)

        register_post_function_hook(
            model=model,
            op_name=param_name,
            port_id=0,
            hook=hook_module,
        )

    return model


def get_pruned_modules(model: nn.Module) -> dict[str, RBPruningMask]:
    """
    Retrieves a mapping of operation names to their corresponding
    RBSparsifyingWeight hooks from the given model.

    :param model: The model from which to retrieve the sparsity modules.
    :return: A dictionary mapping tensor names to their corresponding MagnitudeSparsityBinaryMask instances.
    """
    hook_storage = get_hook_storage(model)
    sparsity_module_map: dict[str, RBPruningMask] = dict()

    for name, hook in hook_storage.named_hooks():
        if not isinstance(hook, RBPruningMask):
            continue

        hook_type, op_name, port_id = decode_hook_name(name)
        if hook_type != "post_hooks" or port_id != 0:
            msg = f"Unexpected place of SparsityBinaryMask: {hook_type=}, {op_name=}, {port_id=}"
            raise nncf.InternalError(msg)
        sparsity_module_map[op_name] = hook

    return sparsity_module_map
