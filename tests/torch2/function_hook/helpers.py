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

from nncf.experimental.torch2.function_hook.wrapper import register_post_function_hook
from nncf.experimental.torch2.function_hook.wrapper import wrap_model


class CallCount(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return x


class AddModule(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))

    def forward(self, x):
        return x + self.w


class ConvModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones([1, 1, 3, 3])

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = torch.relu(x)
        return x


@torch.jit.script
def jit_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.norm(x)


class ConvJitNormModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones([1, 1, 3, 3])

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = jit_norm(x)
        return x


class SimpleModel(nn.Module):

    @staticmethod
    def get_example_inputs():
        return torch.ones([1, 1, 3, 3])

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 1)
        self.simple = ConvModel()

    def forward(self, x):
        x = x + 1
        x += 1
        x = torch.add(input=x, other=1)
        return self.simple(self.conv1(x)) + 1, x


def get_wrapped_simple_model_with_hook() -> ConvModel:
    model = ConvModel()
    wrapped = wrap_model(model)
    register_post_function_hook(wrapped, "conv/conv2d/0", 0, AddModule([2.0]))
    wrapped.eval()
    return wrapped


class ModelMultiEdge(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + x
        return x
