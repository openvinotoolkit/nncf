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

from typing import Callable, Optional, Tuple, TypeVar

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from nncf import Dataset
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv
from tests.torch.helpers import create_depthwise_conv
from tests.torch.helpers import create_transpose_conv
from tests.torch.helpers import set_torch_seed

TTensor = TypeVar("TTensor")


class StaticDatasetMock:
    """
    Common dataset that generate same data and can used for any backend by set fn_to_type function
    to convert data to backend specific type.
    """

    def __init__(
        self,
        input_size: Tuple,
        fn_to_type: Callable = None,
        length: int = 1,
    ):
        super().__init__()
        self._len = length
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
    input_size: Tuple, transform_fn: Callable, fn_to_type: Optional[Callable] = None, length: int = 1
) -> Dataset:
    """
    Create nncf.Dataset for StaticDatasetMock.
    :param input_size: Size of generated tensors,
    :param transform_fn: Function to transformation dataset.
    :param fn_to_type: Function, defaults to None.
    :param length: The length of the dataset.
    :return: Instance of nncf.Dataset for StaticDatasetMock.
    """
    return Dataset(
        StaticDatasetMock(input_size, fn_to_type, length),
        transform_fn,
    )


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


class ConvBiasBNTestModel(torch.nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2)
        self.conv.bias.data = torch.Tensor([0.3, 1.3])
        self.conv.weight.data = torch.Tensor([[[[0.1, -2.0], [1.0, 0.1]]], [[[0.1, 2.0], [-1.0, 0.1]]]])
        self.bn = create_bn(2)
        self.bn.bias.data = torch.Tensor([0.1, 1.0])
        self.bn.weight.data = torch.Tensor([0.2, 2.0])

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CustomConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor([[[[0.1, -2.0], [1.0, 0.1]]], [[[0.1, 2.0], [-1.0, 0.1]]]]))
        self.bias = nn.Parameter(torch.Tensor([0.1, 1.0]))
        self.act = nn.Identity()

    def forward(self, x):
        return self.act(F.conv2d(x, self.weight, self.bias))


class CustomConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = CustomConv()
        self.drop = nn.Dropout(0)

    def forward(self, x):
        return self.drop(self.conv(x))


class CustomBN2d(nn.BatchNorm2d):
    def __init__(self):
        super().__init__(2)
        self.bias.data = torch.Tensor([0.1, 1.0])
        self.weight.data = torch.Tensor([0.2, 2.0])
        self.act = nn.Identity()


class CustomConvBNTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        self.conv = CustomConv()
        self.bn = CustomBN2d()

    def forward(self, x):
        return self.bn(self.conv(x))


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
            self.conv_5 = self._build_conv(3, 2, 1)
            self.max_pool = torch.nn.MaxPool2d((2, 2))
            self.conv_6 = self._build_conv(2, 3, 1)

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
        x = self.conv_5(F.relu(x_1_2))
        x = self.max_pool(x)
        return self.conv_6(x)


class LinearMultiShapeModel(nn.Module):
    INPUT_SIZE = [1, 3, 4, 2]

    def __init__(self) -> None:
        super().__init__()
        with set_torch_seed():
            self.matmul_1_data = torch.randn((4, 4), dtype=torch.float32)
            self.matmul_2_data = torch.randn((4, 4), dtype=torch.float32)
            self.matmul_3_data = torch.randn((1, 8, 2), dtype=torch.float32)
            self.matmul_4_data = torch.randn((1, 8, 3), dtype=torch.float32)
            self.matmul_5_data = torch.randn((1), dtype=torch.float32)
            self.matmul_6_data = torch.randn((8), dtype=torch.float32)

            self.linear_1 = nn.Linear(2, 8)
            self.linear_1.weight.data = torch.randn((8, 2), dtype=torch.float32)
            self.linear_1.bias.data = torch.randn((1, 8), dtype=torch.float32)

            self.linear_2 = nn.Linear(2, 8)
            self.linear_2.weight.data = torch.randn((8, 2), dtype=torch.float32)
            self.linear_2.bias.data = torch.randn((1, 8), dtype=torch.float32)

            self.matmul_7_data = torch.randn((6, 6), dtype=torch.float32)
            self.matmul_8_data = torch.randn((10, 6), dtype=torch.float32)

            self.linear_3 = nn.Linear(4, 4)
            self.linear_3.weight.data = torch.randn((4, 4), dtype=torch.float32)
            self.linear_3.bias.data = torch.randn((1, 4), dtype=torch.float32)

            self.linear_4 = nn.Linear(4, 4)
            self.linear_4.weight.data = torch.randn((4, 4), dtype=torch.float32)
            self.linear_4.bias.data = torch.randn((1, 4), dtype=torch.float32)

    def forward(self, x):
        x = torch.reshape(x, (1, 3, 2, 4))

        x_1 = torch.matmul(x, self.matmul_1_data)
        x_2 = torch.matmul(x, self.matmul_2_data)

        x = torch.add(x_1, x_2)

        x_3 = self.linear_3(x)
        x_4 = self.linear_4(x)

        x_ = torch.add(x_3, x_4)

        x = torch.add(x, x_)
        x = torch.sub(x, x_)

        x_1 = torch.reshape(x, (1, 3, 8))

        x_1_1 = torch.matmul(x_1, self.matmul_3_data)
        x_1_1 = torch.reshape(x_1_1, (1, 6))
        x_1_1 = torch.matmul(self.matmul_5_data, x_1_1)

        x_1_2 = torch.matmul(self.matmul_4_data, x_1)
        x_1_2 = torch.max(x_1_2, 1).values
        x_1_2 = torch.matmul(x_1_2, self.matmul_6_data)

        x_2, x_3 = torch.split(x, 2, 3)
        x_2 = self.linear_1(x_2)
        x_2 = torch.min(x_2, -1).values
        x_2 = torch.flatten(x_2)
        x_2 = torch.matmul(x_2, self.matmul_7_data)
        x_3 = self.linear_2(x_3)
        x_3 = torch.mean(x_3, -1)
        x_3 = torch.flatten(x_3)
        x_3 = torch.matmul(self.matmul_8_data, x_3)
        return x_1_1, x_1_2, x_2, x_3


class NonZeroLinearModel(nn.Module):
    INPUT_SIZE = [10]

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 5)
        self.linear.weight.data = torch.ones((5, 1))
        self.linear.bias.data = torch.zeros((1, 1))

        self.linear1 = nn.Linear(10, 10)
        self.linear1.weight.data = torch.ones((10, 10))
        self.linear1.bias.data = torch.zeros((1, 1))

    def forward(self, x):
        zeros = (x > torch.inf).float()
        empty = torch.nonzero(zeros).reshape((-1, 1, 1)).float()
        y = self.linear(empty)
        y += 5
        y = torch.cat((torch.ones((1, 10)), y.reshape(1, -1)), dim=1)
        y = self.linear1(y)
        y += 5
        return y


class SplittedModel(nn.Module):
    INPUT_SIZE = [1, 2, 28, 28]

    def __init__(self) -> None:
        super().__init__()
        with set_torch_seed():
            self.concat_1_data = torch.randn((1, 1, 28, 28), dtype=torch.float32)
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
        x = torch.concat([self.concat_1_data, x], 1)
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


class EmbeddingModel(nn.Module):
    INPUT_SIZE = [1, 10]
    EMBEDDING_SHAPE = [10, 20]
    MATMUL_W_SHAPE = [5, 20]

    def __init__(self) -> None:
        super().__init__()
        with set_torch_seed():
            self.embedding = nn.Embedding(self.EMBEDDING_SHAPE[0], self.EMBEDDING_SHAPE[1])
            self.embedding.weight.data = torch.randn(self.EMBEDDING_SHAPE, dtype=torch.float32)
            self.matmul = nn.Linear(self.EMBEDDING_SHAPE[1], self.MATMUL_W_SHAPE[1])
            self.matmul.weight.data = torch.randn(self.MATMUL_W_SHAPE, dtype=torch.float32)
            self.matmul.bias.data = torch.randn([1, self.MATMUL_W_SHAPE[0]], dtype=torch.float32)

    def forward(self, x):
        x = x.type(torch.int32)
        x = self.embedding(x)
        x = self.matmul(x)
        return x


class ShareWeghtsConvAndShareLinearModel(nn.Module):
    INPUT_SIZE = [1, 1, 4, 4]

    def __init__(self):
        super().__init__()
        with set_torch_seed():
            self.conv = create_conv(1, 1, 1)
            self.linear = nn.Linear(4, 4)
            self.linear.weight.data = torch.randn((4, 4), dtype=torch.float32)
            self.linear.bias.data = torch.randn((1, 4), dtype=torch.float32)

    def forward(self, x):
        for _ in range(2):
            x = self.conv(x)
            x = self.linear(x)
        return x


class ScaledDotProductAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        return nn.functional.scaled_dot_product_attention(query, key, value)


class DepthwiseConvTestModel(nn.Module):
    INPUT_SIZE = [1, 2, 4, 4]

    def __init__(self):
        super().__init__()
        with set_torch_seed():
            self.conv = create_depthwise_conv(2, 1, 1, 1)
            self.conv.weight.data = torch.randn([2, 1, 1, 1])
            self.conv.bias.data = torch.randn([2])

    def forward(self, x):
        return self.conv(x)


class TransposeConvTestModel(nn.Module):
    INPUT_SIZE = [1, 1, 3, 3]

    def __init__(self):
        super().__init__()
        with set_torch_seed():
            self.conv = create_transpose_conv(1, 2, 2, 1, 1, 2)
            self.conv.weight.data = torch.randn([1, 2, 2, 2])
            self.conv.bias.data = torch.randn([2])

    def forward(self, x):
        return self.conv(x)


class RoPEModel(nn.Module):
    INPUT_SIZE = [1, 10]

    def __init__(self):
        super().__init__()
        with set_torch_seed():
            self.data = torch.randn([5])

    def forward(self, x):
        x = torch.unsqueeze(x, dim=0)
        reshape = torch.reshape(self.data, [1, 5, 1])
        x = torch.matmul(reshape, x)
        x = torch.transpose(x, 2, 1)
        x = torch.cat([x], dim=2)
        x1 = x.sin()
        x2 = x.cos()
        return x1, x2
