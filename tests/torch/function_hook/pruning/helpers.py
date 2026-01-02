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


class ConvModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 3, 3, 3)

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.conv.weight.data = torch.arange(1, 82, dtype=torch.float32).view(3, 3, 3, 3)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = torch.relu(x)
        return x


class MatMulLeft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(3, 3, dtype=torch.float32))

    @staticmethod
    def get_example_inputs():
        return torch.ones(3, 3)

    def forward(self, x: torch.Tensor):
        return torch.matmul(x, self.w)


class MatMulRight(MatMulLeft):
    def forward(self, x: torch.Tensor):
        return torch.matmul(self.w, x)


class SharedParamModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones(3, 3)

    def __init__(self):
        super().__init__()
        shared_linear = nn.Linear(3, 3, bias=False)
        self.module1 = nn.Sequential(shared_linear)
        self.module2 = nn.Sequential(shared_linear)

    def forward(self, x: torch.Tensor):
        return self.module1(x) + self.module2(x)


class TwoConvModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 2, 2, 2)

    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(2, 2, 1, bias=False)

        self.conv1.weight.data = torch.tensor([[[[1.0]], [[-8.0]]], [[[9.0]], [[-10.0]]]])
        self.conv2.weight.data = torch.tensor([[[[1.0]], [[-2.0]]], [[[3.0]], [[-10.0]]]])

    def forward(self, x: torch.Tensor):
        return self.conv2(self.conv1(x))


class MultiDeviceModel(nn.Module):
    @staticmethod
    def get_example_inputs():
        return torch.ones(1, 2, 2, 2)

    def __init__(
        self,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 1, bias=False)
        self.conv2 = nn.Conv2d(2, 2, 1, bias=False).cuda()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.to("cuda")
        x = self.conv2(x)
        return x.cpu()
