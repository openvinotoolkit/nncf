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

from nncf.torch.function_hook.wrapper import register_post_function_hook
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.layer_utils import COMPRESSION_MODULES
from nncf.torch.layer_utils import StatefulModuleInterface


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


class MatMulLeft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([1], dtype=torch.float32))

    @staticmethod
    def get_example_inputs():
        return torch.ones([1, 1])

    def forward(self, x):
        return torch.matmul(x, self.w)


class MatMulRight(MatMulLeft):
    def forward(self, x):
        return torch.matmul(self.w, x)


class QuantizedConvModel(nn.Module):
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


class SharedParamModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones([1, 3])

    def __init__(self):
        super().__init__()
        shared_linear = nn.Linear(3, 1, bias=False)
        self.module1 = nn.Sequential(shared_linear)
        self.module2 = nn.Sequential(shared_linear)

    def forward(self, x):
        return self.module1(x) + self.module2(x)


class CounterHook(nn.Module):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        return x + 1


@COMPRESSION_MODULES.register()
class HookWithState(torch.nn.Module, StatefulModuleInterface):
    def __init__(self, state: str):
        super().__init__()
        self._state = state
        self._dummy_param = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return x + self._dummy_param

    def get_config(self):
        return self._state

    @classmethod
    def from_config(cls, state: str):
        return cls(state)


class ModelGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = torch.nn.GRU(3, 4, batch_first=True)

    def forward(self, x):
        return self.gru(x)
