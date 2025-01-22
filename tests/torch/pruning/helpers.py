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

import copy

import torch
from torch import nn

from nncf.config import NNCFConfig
from nncf.torch.dynamic_graph.context import no_nncf_trace
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv
from tests.torch.helpers import create_depthwise_conv
from tests.torch.helpers import create_grouped_conv
from tests.torch.helpers import create_transpose_conv
from tests.torch.helpers import fill_linear_weight
from tests.torch.test_models.pnasnet import CellB


class PruningTestModel(nn.Module):
    CONV_1_NODE_NAME = "PruningTestModel/NNCFConv2d[conv1]/conv2d_0"
    CONV_2_NODE_NAME = "PruningTestModel/NNCFConv2d[conv2]/conv2d_0"
    CONV_3_NODE_NAME = "PruningTestModel/NNCFConv2d[conv3]/conv2d_0"

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 2, 9, -2)
        self.relu = nn.ReLU()
        self.conv2 = create_conv(3, 1, 3, -10, 0)
        self.conv3 = create_conv(1, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DiffConvsModel(nn.Module):
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


class BranchingModel(nn.Module):
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


class ResidualConnectionModel(nn.Module):
    def __init__(self, last_layer_accept_pruning=True):
        super().__init__()
        self.last_layer_accept_pruning = last_layer_accept_pruning
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
        b, *_ = x.size()
        view_const = (b, -1) if self.last_layer_accept_pruning else (-1)
        x = x.view(view_const)
        x = self.linear(x)
        return x


class EltwiseCombinationModel(nn.Module):
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


class PruningTestModelConcatWithLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 2, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(16, 32, 2, 2, -2)
        for i in range(32):
            self.conv2.weight.data[i] += i
        self.conv3 = create_conv(16, 32, 2, 2, -2)
        for i in range(32):
            self.conv3.weight.data[i] += i
        self.relu = nn.ReLU()
        self.linear = nn.Linear(64 * 6 * 6, 1)
        self.linear.weight.data[0] = 1

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([self.conv2(x), self.conv3(x)], dim=1)
        x = self.relu(x.view(1, -1))
        x = self.linear(x)
        return x


class PruningTestModelDiffChInPruningCluster(nn.Module):
    def __init__(self):
        super().__init__()
        # input_shape=[1, 1, 8, 8]
        self.first_conv = create_conv(1, 16, 2, 1, -2)
        for i in range(16):
            self.first_conv.weight.data[i] += i
        self.conv1 = create_conv(16, 32, 2, 1, -2)
        for i in range(32):
            self.conv1.weight.data[i] += i
        self.linear1 = nn.Linear(16 * 7 * 7, 32 * 6 * 6)
        for i in range(32 * 6 * 6):
            self.linear1.weight.data[i] += i
        self.last_linear = nn.Linear(32 * 6 * 6, 1)
        self.last_linear.weight.data[0] = 1

    def forward(self, x):
        x = self.first_conv(x)
        y = self.conv1(x).view(1, -1)
        z = self.linear1(x.view(1, -1))
        return self.last_linear(y + z)


class PruningTestBatchedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # input_shape=[1, 1, 8, 8]
        self.first_conv = create_conv(1, 32, 1)
        for i in range(32):
            self.first_conv.weight.data[i] += i
        self.linear1 = nn.Linear(8, 16)
        for i in range(16):
            self.linear1.weight.data[i] = i
        self.last_linear = nn.Linear(32 * 8 * 16, 1)
        fill_linear_weight(self.last_linear, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.linear1(x)
        return self.last_linear(x.view(1, -1))


class PruningTestModelBroadcastedLinear(nn.Module):
    def __init__(self):
        super().__init__()
        # input_shape=[1, 1, 8, 8]
        self.first_conv = create_conv(1, 32, 1)
        for i in range(32):
            self.first_conv.weight.data[i] += i
        self.conv1 = create_conv(32, 16, 1)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.linear1 = nn.Linear(32 * 8 * 8, 16)
        for i in range(16):
            self.linear1.weight.data[i] = i
        self.last_linear = nn.Linear(16 * 8 * 8, 1)
        fill_linear_weight(self.last_linear, 1)

    def forward(self, x):
        x = self.first_conv(x)
        y = self.conv1(x)
        z = self.linear1(x.view(1, -1))
        x = y + z.view(1, -1, 1, 1)
        return self.last_linear(x.view(1, -1))


class PruningTestModelBroadcastedLinearWithConcat(nn.Module):
    def __init__(self):
        super().__init__()
        # input_shape=[1, 1, 8, 8]
        self.first_conv = create_conv(1, 32, 1)
        for i in range(32):
            self.first_conv.weight.data[i] += i
        self.conv1 = create_conv(32, 16, 1)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.linear1 = nn.Linear(32 * 8 * 8, 16)
        for i in range(16):
            self.linear1.weight.data[i] = i
        self.conv2 = create_conv(32, 16, 1)
        for i in range(16):
            self.conv2.weight.data[i] += i
        self.last_linear = nn.Linear(2 * 16 * 8 * 8, 1)
        fill_linear_weight(self.last_linear, 1)

    def forward(self, x):
        x = self.first_conv(x)
        y = self.conv1(x)
        z = self.linear1(x.view(1, -1))
        w = y + z.view(1, -1, 1, 1)
        p = self.conv2(x)
        x = torch.cat((w, p), dim=1)
        return self.last_linear(x.view(1, -1))


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


class PruningTestMeanMetatype(nn.Module):
    def __init__(self, mean_dim):
        super().__init__()
        self.mean_dim = mean_dim
        conv2_input_dim = 1 if mean_dim == 1 else 16
        self.conv1 = create_conv(1, 16, 2, 1, -2)
        self.last_conv = create_conv(conv2_input_dim, 32, 1, 2, -2)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.mean(x, self.mean_dim, keepdim=True)
        x = self.last_conv(x)
        return x


class BigPruningTestModel(nn.Module):
    def __init__(self, dim=2):
        super().__init__()
        self.dim = dim
        self.conv1 = create_conv(1, 16, 2, 0, 1, dim=self.dim)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.bn1 = create_bn(16, dim=self.dim)
        self.relu = nn.ReLU()
        self.conv_depthwise = create_depthwise_conv(16, 3, 0, 1, dim=self.dim)
        for i in range(16):
            self.conv_depthwise.weight.data[i] += i
        self.conv2 = create_conv(16, 32, 3, 20, 0, dim=self.dim)
        for i in range(32):
            self.conv2.weight.data[i] += i
        self.bn2 = create_bn(32, dim=self.dim)
        self.up = create_transpose_conv(32, 64, 3, 3, 1, 2, dim=self.dim)
        for i in range(64):
            self.up.weight.data[0][i] += i
        self.linear = nn.Linear(448 * 7 ** (self.dim - 1), 128)
        self.layernorm = nn.LayerNorm(128)
        for i in range(128):
            self.linear.weight.data[i] = i
            self.layernorm.weight.data[i] = i
        self.linear.bias.data.fill_(1)
        self.layernorm.bias.data.fill_(1)
        self.bn3 = create_bn(128, dim=self.dim)
        self.conv3 = create_conv(128, 1, 1, 5, 1, dim=self.dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_depthwise(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.up(x)
        x = self.relu(x)
        b, *_ = x.size()
        x = self.linear(x.view(b, -1))
        x = self.layernorm(x).view(b, -1, *[1] * self.dim)
        x = self.bn3(x)
        x = self.conv3(x)
        x = x.view(b, -1)
        return x


class TestShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, downsample):
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


class ShuffleNetUnitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 16, 1, 1, -2)
        self.unit1 = TestShuffleUnit(16, 16, False)

    def forward(self, x):
        x = self.conv(x)
        x = self.unit1(x)
        return x


class ShuffleNetUnitModelDW(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 16, 1, 1, -2)
        self.unit1 = TestShuffleUnit(16, 32, True)

    def forward(self, x):
        x = self.conv(x)
        x = self.unit1(x)
        return x


class MultipleForwardModel(nn.Module):
    def __init__(self, repeat_seq_of_shared_convs=False, additional_last_shared_layers=False):
        super().__init__()
        self.num_iter_shared_convs = 2 if repeat_seq_of_shared_convs else 1
        self.last_shared_layers = additional_last_shared_layers
        self.conv1 = create_conv(2, 16, 1, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(2, 16, 1, 1, -2)
        for i in range(16):
            self.conv2.weight.data[i] += i
        self.conv3 = create_conv(2, 16, 1, 1, -2)
        # Weights of conv3 is initialized to check difference masks
        self.conv4 = create_conv(16, 16, 1, 1, -2)
        for i in range(16):
            self.conv4.weight.data[i] += i
        self.conv5 = create_conv(16, 16, 1, 1, -2)
        for i in range(16):
            self.conv5.weight.data[i] += i

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        for _ in range(self.num_iter_shared_convs):
            x1 = self.conv4(x1)
            x2 = self.conv4(x2)
            x3 = self.conv4(x3)

        if self.last_shared_layers:
            x1 = self.conv5(x1)
            x2 = self.conv5(x2)
            x3 = self.conv5(x3)
        return x1, x2, x3


class GroupNormModel(nn.Module):
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


class PruningTestModelWrongDims(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = create_conv(1, 8, 1)
        self.last_conv = create_conv(32, 1, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = x.view(-1, 32, 4, 4)
        return self.last_conv(x)


class PruningTestModelWrongDimsElementwise(nn.Module):
    def __init__(self, use_last_conv=True):
        super().__init__()
        self.use_last_conv = use_last_conv
        self.first_conv = create_conv(1, 8, 1)
        self.branch_conv = create_conv(8, 32, 2, stride=2)
        if use_last_conv:
            self.last_conv = create_conv(32, 1, 1)

    def forward(self, x):
        x = self.first_conv(x)
        y = self.branch_conv(x)
        x = x.view(y.shape)
        x = x + y
        if self.use_last_conv:
            x = self.last_conv(x)
        return x


class PruningTestModelSimplePrunableLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 4, 1)
        self.linear = nn.Linear(256, 32)
        self.last_linear = nn.Linear(32, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(torch.flatten(x, start_dim=1))
        x = self.last_linear(x)
        return x


class DisconectedGraphModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 1, 1, -2)
        for i in range(16):
            self.conv1.weight.data[i] += i
        self.conv2 = create_conv(16, 16, 1, 1, -2)
        for i in range(16):
            self.conv2.weight.data[i] += i
        self.conv3 = create_conv(16, 1, 1, 1, -2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv1(x)
        # Broke tracing graph by no_nncf_trace
        with no_nncf_trace():
            x = self.relu(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64)
        x = self.fc(x)
        # Broke tracing graph by no_nncf_trace
        with no_nncf_trace():
            x = self.relu(x)
        x = copy.copy(x)
        return x


class DepthwiseConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 512, 1, 1, 1)
        self.conv4 = create_conv(1024, 512, 2, 1, 1)
        for i in range(512):
            self.conv1.weight.data[i] += i
            self.conv4.weight.data[i] += i
        self.conv2 = create_conv(512, 1024, 3, 1, 1)
        self.conv3 = create_conv(512, 1024, 3, 1, 1)
        self.depthwise_conv = create_depthwise_conv(1024, 5, 1, 1)
        for i in range(1024):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
            self.depthwise_conv.weight.data[i] += i

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x = x1 + x2
        x = self.depthwise_conv(x)
        return self.conv4(x)


class MultipleDepthwiseConvolutionModel(nn.Module):
    """
    Model with group conv which has
    out_channels = 2 * in_channels and
    in_channels == groups
    """

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 512, 1, 1, 1)
        self.conv4 = create_conv(2048, 512, 2, 1, 1)
        for i in range(512):
            self.conv1.weight.data[i] += i
            self.conv4.weight.data[i] += i
        self.conv2 = create_conv(512, 1024, 3, 1, 1)
        self.conv3 = create_conv(512, 1024, 3, 1, 1)
        self.depthwise_conv = create_grouped_conv(1024, 2048, 5, 1024, 1, 1)
        for i in range(1024):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
            self.depthwise_conv.weight.data[i] += i

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x)
        x = x1 + x2
        x = self.depthwise_conv(x)
        return self.conv4(x)


class GroupedConvolutionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 512, 1, 1, 1)
        for i in range(512):
            self.conv1.weight.data[i] += i
        self.conv2 = create_grouped_conv(512, 128, 3, 4, 1, 1)
        self.conv3 = create_depthwise_conv(128, 3, 1, 1)
        for i in range(128):
            self.conv2.weight.data[i] += i
            self.conv3.weight.data[i] += i
        self.fc = nn.Linear(2048, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1)
        return self.fc(x)


class SplitIdentityModel(nn.Module):
    #         (input)
    #            |
    #         (conv1)
    #            |
    #         (chunk)
    #            |
    #         (conv2)
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 4, 1, 1)
        self.conv2 = create_conv(4, 8, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = torch.chunk(x, chunks=1, dim=1)
        y1 = self.conv2(y1[0])
        return y1


class SplitMaskPropFailModel(nn.Module):
    #         (input)
    #            |
    #         (conv1)
    #            |
    #         (chunk)
    #        /      \
    #    (conv2)  (conv3)
    """
    Weights have shape [N, C, C, W] and split dimension is not 1, but 2.
    Mask propagation should fail because of inconsistency of number of channels (C)
    and length of the resulting mask (C/2).
    """

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 6, 3, 1)
        self.conv2 = create_conv(6, 12, 1, 1)
        self.conv3 = create_conv(6, 12, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1, y2 = torch.chunk(x, chunks=2, dim=2)

        y1 = self.conv2(y1)
        y2 = self.conv3(y2)
        return y1, y2


class SplitPruningInvalidModel(nn.Module):
    #         (input)
    #            |
    #         (conv1)
    #            |
    #         (chunk)
    #        /      \
    #    (conv2)  (conv3)
    """
    Weights have shape [N, C, 2C, W] and split dimension is not 1, but 2.
    Mask propagation won't fail with the current code, but pruning will be invalid.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 3, 1)
        self.conv2 = create_conv(3, 6, 1, 1)
        self.conv3 = create_conv(3, 6, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1, y2 = torch.chunk(x, chunks=2, dim=2)

        y1 = self.conv2(y1)
        y2 = self.conv3(y2)
        return y1, y2


class SplitConcatModel(nn.Module):
    #         (input)
    #            |
    #         (conv1)
    #            |
    #         (chunk)
    #        /      \
    #    (conv2)  (conv3)
    #         \    /
    #        (concat)
    #           |
    #        (conv4)
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 1, 1)
        self.conv2 = create_conv(8, 16, 1, 1)
        self.conv3 = create_conv(8, 16, 1, 1)
        self.conv4 = create_conv(32, 64, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1, y2 = torch.chunk(x, chunks=2, dim=1)

        y1 = self.conv2(y1)
        y2 = self.conv3(y2)

        y = torch.cat([y1, y2], dim=1)
        y = self.conv4(y)
        return y


class MultipleSplitConcatModel(nn.Module):
    #              (input)
    #                 |
    #              (conv1)
    #             /      \
    #        (chunk)    (chunk)
    #        /     \  /   \    \
    #  (conv2) (concat) (conv3) (conv4)
    #         \    /
    #        (concat)
    #           |
    #        (conv5)
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 1, 1)
        self.conv2 = create_conv(8, 16, 1, 1)
        self.conv3 = create_conv(6, 12, 1, 1)
        self.conv4 = create_conv(4, 8, 1, 1)
        self.conv5 = create_conv(14, 28, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1, y2 = torch.chunk(x, chunks=2, dim=1)
        y1 = self.conv2(y1)

        y3, y4, y5 = torch.chunk(x, chunks=3, dim=1)
        y = torch.cat([y2, y3], dim=1)

        y4 = self.conv3(y4)
        y5 = self.conv4(y5)
        y = self.conv5(y)
        return y


class SplitReshapeModel(nn.Module):
    #         (input)
    #            |
    #         (conv1)
    #            |
    #         (chunk)
    #        /      \
    #  (reshape1) (reshape2)
    #      |          |
    #   (conv2)    (conv3)
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 8, 1, 1)
        self.conv2 = create_conv(4, 8, 1, 1)
        self.conv3 = create_conv(4, 8, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1, y2 = torch.chunk(x, chunks=2, dim=2)
        y1 = torch.reshape(y1, [1, 4, 8, 8])
        y2 = torch.reshape(y2, [1, 4, 8, 8])
        y1 = self.conv2(y1)
        y2 = self.conv3(y2)
        return y1, y2


class HRNetBlock(nn.Module):
    # omit interpolate, min
    #                    (input)
    #                   /      \
    #             (conv1)     (conv2)
    #               /            |
    #            (chunk1)      (chunk2)
    #            /    \       /     \
    #           /      \  (pool) (conv3)
    #          /        \   /       /
    #      (conv4)     (concat1)   /
    #        \           /        /
    #         \      (conv5)     /
    #          \    /     \     /
    #        (concat2)   (concat3)
    #           |            |
    #        (conv6)     (conv7)
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 16, 5, 5)
        self.conv2 = create_conv(1, 16, 1, 1)
        self.conv3 = create_conv(8, 16, 5, 5)
        self.conv4 = create_conv(8, 16, 1, 1)
        self.conv5 = create_conv(16, 32, 1, 1)
        self.conv6 = create_conv(48, 48, 1, 1)
        self.conv7 = create_conv(48, 48, 1, 1)
        for conv in [getattr(self, f"conv{i}") for i in range(1, 8)]:
            for i in range(conv.out_channels):
                conv.weight.data[i] = i

        self.avg_pool = nn.AdaptiveAvgPool2d(4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1, x1 = torch.chunk(x1, chunks=2, dim=1)  # chunk1
        x2, y2 = torch.chunk(x2, chunks=2, dim=1)  # chunk2
        x2 = self.avg_pool(x2)
        y2 = self.conv3(y2)
        y1 = self.conv4(y1)
        y = torch.cat([x1, x2], dim=1)  # concat1
        y = self.conv5(y)
        out1 = torch.cat([y1, y], dim=1)  # concat2
        out2 = torch.cat([y, y2], dim=1)  # concat3
        out1 = self.conv6(out1)
        out2 = self.conv7(out2)
        return out1, out2


class PruningTestModelPad(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 8, 3, padding=1)
        self.conv2 = create_conv(8, 8, 3, padding=0)
        self.conv3 = create_conv(8, 8, 3, padding=0)
        self.conv4 = create_conv(8, 8, 3, padding=1)
        self.conv5 = create_conv(8, 8, 3, padding=0)
        self.conv6 = create_conv(8, 8, 3, padding=0)
        self.conv7 = create_conv(8, 8, 3, padding=0)
        self.conv8 = create_conv(8, 8, 3, padding=0)
        self.conv9 = create_conv(8, 8, 3, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1))
        x = self.conv3(x)
        x = self.conv4(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant")
        x = self.conv5(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", value=0)
        x = self.conv6(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "reflect")
        x = self.conv7(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), "constant", value=1)
        x = self.conv8(x)
        x = torch.nn.functional.pad(x, (1, 1, 1, 1), value=1)
        x = self.conv9(x)
        return x


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayerWithReshape(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, make_divisible(channel // reduction, 8), 1),
            nn.BatchNorm2d(make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Conv2d(make_divisible(channel // reduction, 8), channel, 1),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = y.view(b, c).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y


class SELayerWithReshapeAndLinear(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, make_divisible(channel // reduction, 8)),
            nn.BatchNorm1d(make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(make_divisible(channel // reduction, 8), channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class SELayerWithReshapeAndLinearAndMean(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(make_divisible(channel // reduction, 8), channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean((2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, se_layer):
        super().__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(
                hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim, bias=False
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            # Squeeze-and-Excite
            se_layer(hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        return self.conv(x)


class MobilenetV3BlockSEReshape(nn.Module):
    def __init__(self, mode="default"):
        super().__init__()
        se_block_map = {
            "default": SELayerWithReshape,
            "linear": SELayerWithReshapeAndLinear,
            "linear_mean": SELayerWithReshapeAndLinearAndMean,
        }

        se_block = se_block_map[mode]
        self.first_conv = nn.Conv2d(1, 6, 2)
        self.inverted_residual = InvertedResidual(6, 6, 6, 5, 1, se_block)
        self.last_conv = nn.Conv2d(6, 1, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.inverted_residual(x)
        return self.last_conv(x)


class NASnetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_conv = nn.Conv2d(1, 2, (2, 2))
        self.cell = CellB(2, 4, 2)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.cell(x)
        return x


def get_basic_pruning_config(input_sample_size=None) -> NNCFConfig:
    if input_sample_size is None:
        input_sample_size = [1, 1, 4, 4]
    config = NNCFConfig()
    config.update(
        {
            "model": "pruning_conv_model",
            "input_info": {
                "sample_size": input_sample_size,
            },
            "compression": {"params": {}},
        }
    )
    return config


def get_pruning_baseline_config(input_sample_size=None) -> NNCFConfig:
    config = get_basic_pruning_config(input_sample_size)
    # Filling params
    compression_config = config["compression"]
    compression_config["params"]["schedule"] = "baseline"
    compression_config["params"]["num_init_steps"] = 1
    return config


def get_pruning_exponential_config(input_sample_size=None) -> NNCFConfig:
    config = get_basic_pruning_config(input_sample_size)
    # Filling params
    compression_config = config["compression"]
    compression_config["params"]["schedule"] = "exponential_with_bias"
    compression_config["params"]["num_init_steps"] = 1
    compression_config["params"]["pruning_steps"] = 20
    return config


def gen_ref_masks(desc):
    return [torch.tensor([0.0] * zeroes + [1.0] * ones) for zeroes, ones in desc]
