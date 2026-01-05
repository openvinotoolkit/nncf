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

import nncf
from nncf.torch.function_hook.pruning.rb.algo import get_pruned_modules


class RBLoss(nn.Module):
    """
    RBLoss is a custom loss function for pruning layers based on a target sparsity ratio.

    This loss function computes the sparsity loss for the given model by evaluating the
    loss from pruned parameters and adjusting it according to the target pruning ratio.

    :param model: The neural network model to be pruned.
    :param target_ratio: The desired target sparsity ratio.
    :param p: A scaling factor for the loss computation, default is 0.05.
    :param current_ratio: The current sparsity ratio of the model.
    """

    def __init__(self, model: nn.Module, target_ratio: float, p: float = 0.05):
        super().__init__()
        self.model = model
        self.target_ratio = target_ratio
        self.p = p
        self.current_ratio = 0.0

        if target_ratio < 0 or target_ratio >= 1:
            msg = "initial_ratio should be in range [0, 1)."
            raise nncf.InternalError(msg)

    def forward(self) -> torch.Tensor:
        """
        Computes the forward pass for the loss function used in pruning.

        This function calculates the loss based on the number of pruned weights
        in the model. It aggregates the loss from all sparse layers and normalizes
        it based on the target ratio.

        :return: The computed loss value, normalized based on the target ratio.
        """
        num_params: int = 0
        loss = torch.tensor(0)

        pruning_modules = get_pruned_modules(self.model)

        if not pruning_modules:
            msg = "No pruning modules were found in the model."
            raise nncf.InternalError(msg)

        for pruning_module in pruning_modules.values():
            sw_loss = pruning_module.loss()
            num_params = num_params + sw_loss.numel()
            # move tensor to cpu to support multi-device models
            loss = loss + sw_loss.sum().cpu()

        self.current_ratio = 1 - float(loss.detach()) / num_params

        return ((loss / num_params - (1 - self.target_ratio)) / self.p).pow(2)
