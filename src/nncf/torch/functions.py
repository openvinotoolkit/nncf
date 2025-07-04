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


def clamp(x, low, high):
    return torch.max(torch.min(x, high), low)


def logit(x):
    return torch.log(x / (1 - x))


class STRound(torch.autograd.Function):
    @staticmethod
    def symbolic(g, input_, inplace=False):
        return g.op("STRound", input_)

    @staticmethod
    def forward(ctx, input_):
        output = input_.round()
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0]


class STThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, threshold: float = 0.5):
        output = (input_ > threshold).type(input_.dtype)
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs[0], None
