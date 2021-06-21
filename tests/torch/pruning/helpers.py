"""
 Copyright (c) 2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import torch
from torch import nn

from nncf.config import NNCFConfig
from tests.torch.helpers import create_conv
from tests.torch.helpers import create_transpose_conv
from tests.torch.helpers import create_depthwise_conv


class PruningTestModel(nn.Module):
    CONV_1_NODE_NAME = "PruningTestModel/NNCFConv2d[conv1]/conv2d_0"
    CONV_2_NODE_NAME = "PruningTestModel/NNCFConv2d[conv2]/conv2d_0"
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 2, 9, -2)
        self.relu = nn.ReLU()
        self.conv2 = create_conv(3, 1, 3, -10, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class TestModelDiffConvs(nn.Module):
    def __init__(self):
        super().__init__()
        # Usual conv
        self.conv1 = create_conv(1, 32, 2, 9, -2)
        self.relu = nn.ReLU()
        # Depthwise conv
        self.conv2 = nn.Conv2d(32, 32, 1, groups=32)

        # Downsample conv
        self.conv3 = create_conv(32, 32, 3, -10, 0, stride=2)

        # Group conv
        self.conv4 = nn.Conv2d(32, 16, 1, groups=8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class TestModelBranching(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 2, 1, -2)
        self.conv2 = create_conv(1, 3, 2, 2, -2)
        self.conv3 = create_conv(1, 3, 2, 3, -2)
        self.relu = nn.ReLU()
        self.conv4 = create_conv(3, 1, 3, 10, 0)
        self.conv5 = create_conv(3, 1, 3, -10, 0)

    def forward(self, x):
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x) + self.conv5(x)
        return x


class TestModelResidualConnection(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 8, 3, 1, -2, padding=1)
        self.conv2 = create_conv(8, 8, 3, 2, -2, padding=1)
        self.conv3 = create_conv(8, 8, 3, 3, -2, padding=1)
        self.conv4 = create_conv(8, 1, 3, 10, 0, padding=1)
        self.conv5 = create_conv(8, 1, 3, -10, 0, padding=1)
        self.linear = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        x = x + self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x) + self.conv5(x)
        x = self.linear(x.view(-1))
        return x


class TestModelEltwiseCombination(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 8, 3, 1, -2, padding=1)
        self.conv2 = create_conv(8, 8, 3, 2, -2, padding=1)
        self.conv3 = create_conv(8, 8, 3, 3, -2, padding=1)
        self.conv4 = create_conv(8, 8, 3, 10, 0, padding=1)
        self.conv5 = create_conv(8, 1, 3, -10, 0, padding=1)
        self.conv6 = create_conv(8, 1, 3, -10, 0, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = x + self.conv2(x)
        x_1 = x + self.conv3(x)
        x_2 = x + self.conv4(x)
        x = self.conv5(x_1) + self.conv6(x_2)
        return x


class PruningTestModelConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 2, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i

        self.conv2 = create_conv(16, 32, 2, 2, -2)
        self.conv3 = create_conv(16, 32, 2, 2, -2)
        for i in range(32):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.relu = nn.ReLU()
        self.conv4 = create_conv(64, 16, 3, 10, 0)
        for i in range(16):
            self.conv4.weight.data[i] += i

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.conv2(x), self.conv3(x)], dim=1)
        x = self.relu(x)
        x = self.conv4(x)
        return x


class PruningTestModelConcatBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 1, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i

        self.conv2 = create_conv(16, 16, 1, 2, -2)
        self.conv3 = create_conv(16, 16, 1, 2, -2)
        for i in range(16):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.relu = nn.ReLU()
        self.conv4 = create_conv(32, 16, 1, 10, 0)
        for i in range(16):
            self.conv4.weight.data[i] += 16 - i
        self.conv5 = create_conv(48, 16, 1, 10, 0)

        self.bn = nn.BatchNorm2d(16)
        self.bn.bias = torch.nn.Parameter(torch.ones(16))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn1.bias = torch.nn.Parameter(torch.ones(32))

        self.bn2 = nn.BatchNorm2d(48)
        self.bn2.bias = torch.nn.Parameter(torch.ones(48))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = torch.cat([self.conv2(x), self.conv3(x)], dim=1)
        x1 = self.bn1(x)
        x1 = self.conv4(x1)
        x = torch.cat([x1, x], dim=1)
        x = self.bn2(x)
        x = self.conv5(x)
        return x


class PruningTestModelEltwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 2, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i

        self.conv2 = create_conv(16, 32, 2, 2, -2)
        self.conv3 = create_conv(16, 32, 2, 2, -2)
        for i in range(32):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.relu = nn.ReLU()
        self.conv4 = create_conv(32, 16, 3, 10, 0)
        for i in range(16):
            self.conv4.weight.data[i] += i

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x) + self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        return x


class BigPruningTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 2, 0, 1)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.relu = nn.ReLU()
        self.conv2 = create_conv(16, 32, 3, 20, 0)
        for i in range(32):
            self.conv2.weight.data[i] += i
        self.bn = nn.BatchNorm2d(32)
        self.up = create_transpose_conv(32, 64, 3, 3, 1, 2)
        for i in range(64):
            self.up.weight.data[0][i] += i
        self.conv3 = create_conv(64, 1, 5, 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.relu(x)
        x = self.conv3(x)
        # x = x.view(1, -1)
        return x


class TestShuffleUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downsample):
        super().__init__()
        self.downsample = downsample
        mid_channels = out_channels // 2

        self.compress_conv1 = create_conv((in_channels if self.downsample else mid_channels), mid_channels, 1, 1, -2)
        self.dw_conv2 = create_depthwise_conv(mid_channels, 3, 2, -2, padding=1, stride=(2 if self.downsample else 1))
        self.expand_conv3 = create_conv(mid_channels, mid_channels, 1, 1, -2)

        if downsample:
            self.dw_conv4 = create_depthwise_conv(in_channels, 3, 2, -2, padding=1, stride=2)
            self.expand_conv5 = create_conv(in_channels, mid_channels, 1, 1, -2)

        self.activ = nn.ReLU(inplace=True)


    def forward(self, x):
        if self.downsample:
            y1 = self.dw_conv4(x)
            y1 = self.expand_conv5(y1)
            y1 = self.activ(y1)
            x2 = x
        else:
            y1, x2 = torch.chunk(x, chunks=2, dim=1)

        y2 = self.compress_conv1(x2)
        y2 = self.activ(y2)
        y2 = self.dw_conv2(y2)
        y2 = self.expand_conv3(y2)

        y2 = self.activ(y2)
        if not self.downsample:
            y2 = y2 + x2
        x = torch.cat((y1, y2), dim=1)
        return x


class TestModelShuffleNetUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 16, 1, 1, -2)
        self.unit1 = TestShuffleUnit(16, 16, False)

    def forward(self, x):
        x = self.conv(x)
        x = self.unit1(x)
        return x

class TestModelShuffleNetUnitDW(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 16, 1, 1, -2)
        self.unit1 = TestShuffleUnit(16, 32, True)

    def forward(self, x):
        x = self.conv(x)
        x = self.unit1(x)
        return x


class TestModelMultipleForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(2, 16, 1, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(2, 16, 1, 1, -2)
        for i in range(16):
            self.conv2.weight.data[i] += i
        self.conv3 = create_conv(2, 16, 1, 1, -2)
        # Wights of conv3 is initialized to check difference masks
        self.conv4 = create_conv(16, 16, 1, 1, -2)
        for i in range(16):
            self.conv4.weight.data[i] += i

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x1 = self.conv4(x1)
        x2 = self.conv4(x2)
        x3 = self.conv4(x3)

        return x1, x2, x3


class TestModelGroupNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 1, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.gn1 = nn.GroupNorm(16, 16)  # Instance Normalization
        self.conv2 = create_conv(16, 16, 1, 1, -2)
        for i in range(16):
            self.conv2.weight.data[i] += i
        self.gn2 = nn.GroupNorm(2, 16)  # Group Normalization

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.conv2(x)
        x = self.gn2(x)
        return x


class PruningTestWideModelConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 512, 1, 1, 1)
        for i in range(512):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(512, 1024, 1, 1, 1)
        self.conv3 = create_conv(512, 1024, 1, 1, 1)
        for i in range(1024):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.conv4 = create_conv(2048, 2048, 1, 1, 1)
        for i in range(2048):
            self.conv4.weight.data[i] += i

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.conv2(x), self.conv3(x)], dim=1)
        x = self.conv4(x)
        return x


class PruningTestWideModelEltwise(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 512, 1, 1, 1)
        for i in range(512):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(512, 1024, 1, 1, 1)
        self.conv3 = create_conv(512, 1024, 1, 1, 1)
        for i in range(1024):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.conv4 = create_conv(1024, 1024, 1, 1, 1)
        for i in range(1024):
            self.conv4.weight.data[i] += i

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) + self.conv3(x)
        x = self.conv4(x)
        return x


class PruningTestModelSharedConvs(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 512, 1, 1, 1)
        for i in range(512):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(512, 1024, 3, 1, 1)
        self.conv3 = create_conv(1024, 1024, 1, 1, 1)
        for i in range(1024):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, in1):
        in1 = self.conv1(in1)
        in2 = self.maxpool(in1)
        out1 = self.conv2(in1)
        out2 = self.conv2(in2)
        return self.conv3(out1), self.conv3(out2)


def get_basic_pruning_config(input_sample_size=None) -> NNCFConfig:
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update({
        "model": "pruning_conv_model",
        "input_info":
            {
                "sample_size": input_sample_size,
            },
        "compression":
            {
                "params": {
                }
            }
    })
    return config


def get_pruning_baseline_config(input_sample_size=None) -> NNCFConfig:
    config = get_basic_pruning_config(input_sample_size)
    # Filling params
    compression_config = config['compression']
    compression_config['params']["schedule"] = "baseline"
    compression_config['params']["num_init_steps"] = 1
    return config


def get_pruning_exponential_config(input_sample_size=None) -> NNCFConfig:
    config = get_basic_pruning_config(input_sample_size)
    # Filling params
    compression_config = config['compression']
    compression_config['params']["schedule"] = "exponential_with_bias"
    compression_config['params']["num_init_steps"] = 1
    compression_config['params']["pruning_steps"] = 20
    return config


def gen_ref_masks(desc):
    return [torch.tensor([0.0] * zeroes + [1.0] * ones) for zeroes, ones in desc]
