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
import torch.nn.functional as F
from abc import abstractmethod
from tests.torch.helpers import create_conv
from torch import nn
from torch.nn import Dropout
from torch.nn import Parameter

from nncf.torch import register_module


class ModelWithDummyParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_param = Parameter(torch.zeros(1))

    @abstractmethod
    def forward(self, x):
        pass


class ManyNonEvalModules(ModelWithDummyParameter):
    class AuxBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.weight = Parameter(torch.ones([1, 1]))

        def forward(self, x):
            x = F.linear(x, self.weight)
            x = self.linear(x)
            x = F.relu(x)
            return x

    @register_module()
    class CustomWeightModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = Parameter(torch.ones([1, 1]))

        def forward(self, x):
            x = F.linear(x, self.weight)
            return x

    class ModuleWithMixedModules(nn.Module):
        def __init__(self):
            super().__init__()
            self.custom = ManyNonEvalModules.CustomWeightModule()
            self.not_called_linear = nn.Linear(1, 1)
            self.called_linear = nn.Linear(1, 1)

        def forward(self, x):
            x = Dropout(p=0.2)(x)
            x = self.custom(x)
            x = Dropout(p=0.2)(x)
            x = self.called_linear(x)
            return x

    def __init__(self):
        super().__init__()
        self.aux_branch = self.AuxBranch()
        self.mixed_modules = self.ModuleWithMixedModules()
        self.avg_pool = nn.AvgPool2d(1)

    def forward(self, x):
        x = self.avg_pool(x)
        if self.training:
            aux = self.aux_branch(x)
        x = self.mixed_modules(x)
        return (x, aux) if self.training else x


class PoolUnPool(ModelWithDummyParameter):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool3d(3, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool3d(3, stride=2)

    def forward(self, input_):
        output, indices = self.pool(input_)
        return self.unpool(output, indices)


class ArangeModel(ModelWithDummyParameter):
    def forward(self, dummy_x):
        return torch.arange(0, dummy_x.size(0), dtype=torch.int64)


class TransposeModel(ModelWithDummyParameter):
    def forward(self, x):
        o1 = x.transpose(dim0=0, dim1=0)
        o2 = x.permute(dims=[0])
        return o1, o2


class GatherModel(ModelWithDummyParameter):
    def forward(self, x):
        index = torch.zeros(1, dtype=torch.int64).to(x.device)
        o1 = torch.where(self.dummy_param > 0, x, self.dummy_param)
        o2 = torch.index_select(x, dim=0, index=index)
        o3 = x.index_select(dim=0, index=index)
        o4 = x[0]
        return o1, o2, o3, o4


class MaskedFillModel(ModelWithDummyParameter):
    def forward(self, x):
        o1 = x.masked_fill_(self.dummy_param > 0, 1.0)
        o2 = x.masked_fill(self.dummy_param > 0, 1.0)
        return o1, o2


class ReshapeModel(ModelWithDummyParameter):
    def forward(self, x):
        torch.squeeze(x)
        torch.unsqueeze(x, dim=0)
        torch.flatten(x)
        return x.reshape([1]), x.squeeze(), x.flatten(), x.unsqueeze(dim=0), x.view([1])


class MultiBranchesModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding=1, groups=3)
        self.max_pool_b = nn.MaxPool2d(kernel_size=3)
        self.conv_b = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=5, padding=3)
        self.conv_c = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=3)
        self.conv_d = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=0)

    def forward(self, x):
        x = nn.ReLU()(x)
        xa = self.conv_a(x)
        xb = self.conv_b(self.max_pool_b(x))
        xc = self.conv_c(x)
        xd = self.conv_d(x)
        return xa, xb, xc, xd


class EmbeddingSumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 10)
        self.embeddingbag = nn.EmbeddingBag(10, 10)

    def forward(self, x):
        y1 = self.embedding(x)
        y2 = self.embeddingbag(x)
        return y1 + y2


class EmbeddingCatLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding1 = nn.Embedding(10, 10)
        self.embedding2 = nn.Embedding(10, 10)
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        y1 = self.embedding1(x)
        y2 = self.embedding2(x)
        z = torch.cat([y1, y2])
        return self.linear(z)

class MultiOutputSameTensorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_param = torch.nn.Parameter(torch.ones([1]))

    def forward(self, x):
        return x, x*x, x


#       fq_2
#        \
# fq_2 - conv_1 - fq_6
#                   \
#        fq_4       add
#         \         /
# fq_4 - conv_2 - fq_6
#
class AddTwoConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 2, 2, -1, -2)
        self.conv2 = create_conv(1, 2, 2, -1, -2)

    def forward(self, x):
        return self.conv1(x) + self.conv2(x)
