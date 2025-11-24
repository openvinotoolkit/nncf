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


from typing import Any

import torch
from torch import Tensor
from torch import nn
from torch.overrides import handle_torch_function
from torch.overrides import has_torch_function_unary

from nncf.torch.functions import STThreshold
from nncf.torch.functions import logit
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import StatefulModuleInterface


def binary_mask(mask: Tensor) -> Tensor:
    """
    Applies a binary mask to the input tensor using a sigmoid function followed by a
    custom thresholding operation.

    :param mask: The input tensor to which the binary mask will be applied.
    :return: The resulting tensor after applying the binary mask.
    """
    return STThreshold.apply(torch.sigmoid(mask))  # type: ignore


def apply_rb_binary_mask(input_: torch.Tensor, mask: torch.Tensor, training: bool, eps: float = 1e-6) -> torch.Tensor:
    """
    Applies a binary mask to the input tensor during training or inference.

    This function modifies the input tensor based on the provided binary mask.
    During training, it adds noise to the mask to encourage robustness.
    In inference mode, it simply applies the mask to the input.

    :param input_: The input tensor to which the mask will be applied.
    :param mask: The binary mask tensor that determines which elements of the input are retained.
    :param training: A flag indicating whether the model is in training mode or not.
    :param eps: A small value to avoid numerical instability when computing the logit.
    :return: The masked input tensor.
    """
    if has_torch_function_unary(input_):
        return handle_torch_function(apply_rb_binary_mask, (input_,), input_, mask, training)  # type: ignore[no-any-return]

    if training:
        uniform = torch.empty_like(mask, requires_grad=False).uniform_()
        mask = mask + logit(uniform.clamp(eps, 1 - eps))

    return input_ * binary_mask(mask)


@COMPRESSION_MODULES.register()
class RBPruningMask(nn.Module, StatefulModuleInterface):
    """
    A module that applies a binary pruning mask to the input tensor.

    This class is designed to create a learnable binary mask for pruning
    operations in neural networks. The mask is initialized with small values
    and can be optimized during training to determine which weights to prune.

    :param mask: A learnable parameter representing the binary mask.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        super().__init__()
        self.mask = nn.Parameter(logit(torch.ones(shape) * 0.99))

    def forward(self, x: Tensor) -> Tensor:
        return apply_rb_binary_mask(x, self.mask, self.training)

    def loss(self) -> Tensor:
        return binary_mask(self.mask)

    def get_config(self) -> dict[str, Any]:
        return {"shape": tuple(self.mask.shape)}

    @classmethod
    def from_config(cls, state: dict[str, Any]) -> "RBPruningMask":
        return RBPruningMask(shape=state["shape"])
