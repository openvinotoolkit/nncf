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


def clamp(x: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    return torch.max(torch.min(x, high), low)


def logit(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


class STThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, input_: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        output = (input_ > threshold).type(input_.dtype)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_outputs[0], None
