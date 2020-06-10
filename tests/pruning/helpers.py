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
from tests.helpers import create_conv


class PruningTestModel(nn.Module):
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


class PruningTestModelBranching(nn.Module):
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

        self.conv3 = create_conv(32, 1, 5, 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = x.view(1, -1)
        return x


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
