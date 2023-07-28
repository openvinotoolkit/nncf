# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Tuple, TypeVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nncf import Dataset
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv
from tests.torch.helpers import set_torch_seed

TTensor = TypeVar("TTensor")


class StaticDatasetMock:
    """
    Common dataset that generate same data and can used for any backend by set fn_to_type function
    to convert data to backend specific type.
    """

    def __init__(self, input_size: Tuple, fn_to_type: Callable = None):
        super().__init__()
        self._len = 1
        self._input_size = input_size
        self._fn_to_type = fn_to_type

    def __getitem__(self, _) -> Tuple[TTensor, int]:
        np.random.seed(0)
        data = np.random.rand(*tuple(self._input_size)).astype(np.float32)
        if self._fn_to_type:
            data = self._fn_to_type(data)
        return data, 0

    def __len__(self) -> int:
        return self._len


def get_static_dataset(
    input_size: Tuple,
    transform_fn: Callable,
    fn_to_type: Callable,
) -> Dataset:
    """
    Create nncf.Dataset for StaticDatasetMock.
    :param input_size: Size of generated tensors,
    :param transform_fn: Function to transformation dataset.
    :param fn_to_type: Function, defaults to None.
    :return: Instance of nncf.Dataset for StaticDatasetMock.
    """
    return Dataset(StaticDatasetMock(input_size, fn_to_type), transform_fn)


class ConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2, -1, -2)
        self.conv.weight.data = torch.Tensor([[[[0.1, -2.0], [1.0, 0.1]]], [[[0.1, 2.0], [-1.0, 0.1]]]])
        self.conv.bias.data = torch.Tensor([0.1, 1.0])

    def forward(self, x):
        return self.conv(x)


class ConvBNTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2, bias=False)
        self.conv.weight.data = torch.Tensor([[[[0.1, -2.0], [1.0, 0.1]]], [[[0.1, 2.0], [-1.0, 0.1]]]])
        self.bn = create_bn(2)
        self.bn.bias.data = torch.Tensor([0.1, 1.0])
        self.bn.weight.data = torch.Tensor([0.2, 2.0])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class FCTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.fc.weight.data = torch.Tensor([[0.1, 0.2, 0.3, 0.2], [0.3, -0.1, 0.2, 0.4]])
        self.fc.bias.data = torch.Tensor([1.0, 1.1])

    def forward(self, x):
        x = self.fc(x)
        return x


class MultipleConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        with set_torch_seed():
            self.conv_1 = self._build_conv(1, 2, 2)
            self.conv_2 = self._build_conv(2, 3, 2)
            self.conv_3 = self._build_conv(1, 2, 3)
            self.conv_4 = self._build_conv(2, 3, 1)
            self.conv_5 = self._build_conv(3, 2, 2)

    def _build_conv(self, in_channels=1, out_channels=2, kernel_size=2):
        conv = create_conv(in_channels, out_channels, kernel_size)
        conv.weight.data = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        conv.bias.data = torch.randn([out_channels])
        return conv

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.conv_2(F.relu(x_1))
        x_2 = self.conv_3(x)
        x_2 = self.conv_4(F.relu(x_2))
        x_1_2 = torch.concat([x_1, x_2])
        return self.conv_5(F.relu(x_1_2))


class LinearModel(nn.Module):
    INPUT_SIZE = [1, 3, 4, 2]
    RESHAPE_SHAPE = (1, 3, 2, 4)
    MATMUL_W_SHAPE = (4, 5)

    def __init__(self) -> None:
        super().__init__()
        with set_torch_seed():
            self.matmul_data = torch.randn(self.MATMUL_W_SHAPE, dtype=torch.float32) - torch.tensor(0.5)
            self.add_data = torch.randn(self.RESHAPE_SHAPE, dtype=torch.float32)

    def forward(self, x):
        x = torch.reshape(x, self.RESHAPE_SHAPE)
        x_1 = torch.matmul(x, self.matmul_data)
        x_2 = torch.add(x, self.add_data)
        return x_1, x_2


class SplittedModel(nn.Module):
    INPUT_SIZE = [1, 3, 28, 28]

    def __init__(self) -> None:
        super().__init__()
        with set_torch_seed():
            self.conv_1 = self._build_conv(3, 12, 3)
            self.add_1_data = torch.randn((1, 12, 26, 26), dtype=torch.float32)
            self.maxpool_1 = torch.nn.MaxPool2d(1)

            self.conv_2 = self._build_conv(12, 18, 1)
            self.conv_3 = self._build_conv(18, 12, 1)

            self.conv_4 = self._build_conv(6, 12, 1)
            self.conv_5 = self._build_conv(12, 18, 3)
            self.add_2_data = torch.randn((1, 18, 24, 24), dtype=torch.float32)
            self.conv_6 = self._build_conv(6, 18, 3)

            self.conv_7 = self._build_conv(36, 48, 1)
            self.add_3_data = torch.randn((1, 36, 24, 24), dtype=torch.float32)
            self.conv_8 = self._build_conv(48, 24, 3)
            self.conv_9 = self._build_conv(36, 24, 3)
            self.conv_10 = self._build_conv(36, 24, 3)

            self.conv_11 = self._build_conv(72, 48, 5)
            self.matmul_1_data = torch.randn((96, 48), dtype=torch.float32)
            self.add_4_data = torch.randn((1, 1, 324), dtype=torch.float32)

            self.linear = nn.Linear(324, 48)
            self.linear.weight.data = torch.randn((48, 324), dtype=torch.float32)
            self.linear.bias.data = torch.randn((1, 48), dtype=torch.float32)

            self.add_5_data = torch.randn((1, 1, 324), dtype=torch.float32)
            self.conv_12 = self._build_conv(96, 18, 3)

            self.linear_2 = nn.Linear(48, 10)
            self.linear_2.weight.data = torch.randn((10, 48), dtype=torch.float32)
            self.linear_2.bias.data = torch.randn((1, 10), dtype=torch.float32)

    def _build_conv(self, in_channels=1, out_channels=2, kernel_size=2):
        conv = create_conv(in_channels, out_channels, kernel_size)
        conv.weight.data = torch.randn([out_channels, in_channels, kernel_size, kernel_size])
        conv.bias.data = torch.randn([out_channels])
        return conv

    def forward(self, x):
        x = self.conv_1(x)
        x = F.relu(x)
        x = torch.add(x, self.add_1_data)
        x = self.maxpool_1(x)

        x_1 = self.conv_2(x)
        x_1 = F.relu(x_1)
        x_1 = self.conv_3(x_1)
        x = torch.add(x, x_1)

        x_1, x_2 = torch.split(x, 6, 1)
        x_1 = self.conv_4(x_1)
        x_1 = F.relu(x_1)
        x_1 = self.conv_5(x_1)
        x_1 = torch.add(x_1, self.add_2_data)

        x_2 = self.conv_6(x_2)

        x = torch.concat([x_1, x_2], 1)
        x_1 = self.conv_7(x)
        x_1 = self.conv_8(x_1)

        x_2 = torch.add(x, self.add_3_data)
        x_2 = self.conv_9(x_2)

        x_3 = self.conv_10(x)

        x = torch.concat([x_1, x_2, x_3], 1)
        x = self.conv_11(x)
        x = torch.reshape(x, [1, 48, 324])
        x = torch.matmul(self.matmul_1_data, x)
        x = torch.add(x, self.add_4_data)

        x_1 = self.linear(x)
        x_2 = torch.reshape(x, [1, 96, 18, 18])
        x_2 = self.conv_12(x_2)
        x_2 = torch.reshape(x_2, [1, 96, 48])

        x = torch.add(x_1, x_2)
        x = self.linear_2(x)

        return torch.flatten(x, 1, 2)
