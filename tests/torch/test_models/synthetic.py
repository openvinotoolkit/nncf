"""
 Copyright (c) 2022 Intel Corporation
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

from torch.nn import BatchNorm2d

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

    def forward(self, x):
        output, indices = self.pool(x)
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


class PartlyNonDifferentialOutputsModel(nn.Module):
    def __init__(self, input_size=None):
        super().__init__()
        self.input_size = [1, 1, 4, 4] if input_size is None else input_size
        self.conv1 = torch.nn.Conv2d(in_channels=self.input_size[1], out_channels=1, kernel_size=3)
        self.conv2_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.conv2_2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, x):
        # first and seconds outputs with requires_grad=True
        # third output with requires_grad = False
        xa = self.conv1(x)
        xb = self.conv2_1(xa)
        with torch.no_grad():
            xc = self.conv2_2(xa)
        return xa, xb, xc


class ContainersOutputsModel(nn.Module):
    def __init__(self, input_size=None):
        super().__init__()
        self.input_size = [1, 1, 4, 4] if input_size is None else input_size
        self.conv1 = torch.nn.Conv2d(in_channels=self.input_size[1], out_channels=1, kernel_size=3)
        self.conv2_1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        self.conv2_2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3)

    def forward(self, x):
        xa = self.conv1(x)
        xb = self.conv2_1(xa)
        xc = self.conv2_2(xa)
        return {"xa": xa, "xb_and_xc": (xb, xc)}


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


class MatMulDivConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.matmul(x, y) / 2
        return self.conv(z)


class MMDivConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 1, 1, 2)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        z = torch.mm(x, y) / 2
        z = z.unsqueeze(0)
        z = z.unsqueeze(0)
        return self.conv(z)


class ConvRelu6HSwishHSigmoid(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 2, 2, 2)
        self.conv2 = create_conv(2, 2, 2, 2)

    @staticmethod
    def _hswish(x: torch.Tensor) -> torch.Tensor:
        return x * torch.nn.functional.relu6(x + 3) / 6

    @staticmethod
    def _hsigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu6(x + 3) / 6

    def forward(self, x: torch.Tensor):
        z = self.conv1(x)
        z = self._hswish(z)
        z = self.conv2(z)
        z = self._hsigmoid(z)
        return z


class ConvGeluGetItem(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 8)
        self.dp = nn.Dropout()
        self.conv1 = nn.Conv1d(8, 8, kernel_size=3, padding=2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dp(x)
        x1 = x.transpose(2, 1)
        x1 = self.conv1(x1)
        x1 = F.gelu(x1[:, :, :-2])

        return x + x1.transpose(2, 1)


class ConvBNLeakyReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 2, 2, 2)
        self.bn = BatchNorm2d(2)

    def forward(self, x: torch.Tensor):
        z = self.conv(x)
        z = self.bn(z)
        z = torch.nn.functional.leaky_relu(z)
        return z

class FC_ConstMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(6, 6)
        self.dp = nn.Dropout()

    def forward(self, x):
        x = self.dp(x)
        x1 = self.fc1(x)
        x1 = x1 * 2
        return x + x1
