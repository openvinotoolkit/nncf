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
from torch.overrides import handle_torch_function
from torch.overrides import has_torch_function_unary

from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import StatefulModuleInterface


def apply_magnitude_sparsity_binary_mask(input_: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Applies a binary mask to the input tensor, effectively zeroing out elements
    of the input tensor where the mask is zero.

    :param input_: The input tensor to which the mask will be applied.
    :param mask: A binary mask tensor of the same shape as input_ where elements are either 0 or 1.

    :return: The input tensor with the mask applied.
    """
    if has_torch_function_unary(input_):
        return handle_torch_function(apply_magnitude_sparsity_binary_mask, (input_,), input_, mask)  # type: ignore[no-any-return]
    return input_ * mask


@COMPRESSION_MODULES.register()
class UnstructuredPruningMask(nn.Module, StatefulModuleInterface):
    """
    A module that applies a binary mask for magnitude-based sparsity.

    This module creates a binary mask that can be used to zero out certain
    elements of a tensor based on their magnitudes.

    :param binary_mask: A tensor that represents the binary mask.
    """

    binary_mask: torch.Tensor

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.register_buffer("binary_mask", torch.ones(shape, dtype=torch.bool))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_magnitude_sparsity_binary_mask(x, self.binary_mask)

    def get_config(self) -> dict[str, Any]:
        return {"shape": tuple(self.binary_mask.shape)}

    @classmethod
    def from_config(cls, state: dict[str, Any]) -> "UnstructuredPruningMask":
        return UnstructuredPruningMask(shape=state["shape"])
