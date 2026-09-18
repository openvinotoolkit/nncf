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
from typing import Any

import torch
from torch import nn

from nncf.torch.function_hook.pruning.magnitude.modules import apply_magnitude_binary_mask
from nncf.torch.layer_utils import StatefulModuleInterface


class StructuredPruningMask(nn.Module, StatefulModuleInterface):
    """
    A module that applies a binary mask for structured magnitude-based pruning with a fixed M:N pattern.

    The mask has the same shape as the pruned weight tensor and is computed so that every consecutive
    group of N elements along the last dimension of the tensor contains exactly M zeroed elements,
    where the zeroed elements are selected by magnitude.

    For the 2:4 pattern this means that every group of four weights contains two zeros and two retained
    values, which is the layout expected by runtimes and hardware that support the 2:4 sparsity format.

    :param shape: Shape of the weight tensor to be masked.
    """

    binary_mask: torch.Tensor

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.register_buffer("binary_mask", torch.ones(shape, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_magnitude_binary_mask(x, self.binary_mask)

    def get_config(self) -> dict[str, Any]:
        return {"shape": tuple(self.binary_mask.shape)}

    @classmethod
    def from_config(cls, state: dict[str, Any]) -> "StructuredPruningMask":
        return StructuredPruningMask(shape=state["shape"])
