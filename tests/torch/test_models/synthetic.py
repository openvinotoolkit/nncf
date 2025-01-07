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
from abc import abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm2d
from torch.nn import Dropout
from torch.nn import Parameter
from torchvision.transforms.functional import normalize

from nncf.torch import nncf_model_input
from nncf.torch import register_module
from nncf.torch.dynamic_graph.io_handling import wrap_nncf_model_outputs_with_objwalk
from tests.torch.helpers import create_bn
from tests.torch.helpers import create_conv


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
        return x, x * x, x


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
        self.relu6 = torch.nn.ReLU6()

    def _hswish(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.relu6(x + 3) / 6

    def _hsigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu6(x + 3) / 6

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


class Baddbmm(torch.nn.Module):
    def forward(self, x, y, z):
        return torch.baddbmm(x, y, z)


class ScaledDotProductModel(nn.Module):
    EMBED_DIM = 4
    INPUT_SIZES = [2, 1, EMBED_DIM]

    def forward(self, x):
        shape = x.shape
        x = x.view(-1).view(shape)
        return nn.functional.scaled_dot_product_attention(x, x, x)


class MHA_single_input(torch.nn.Module):
    EMBED_DIM = 4
    INPUT_SIZES = [2, 1, EMBED_DIM]

    def __init__(self):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=self.EMBED_DIM, num_heads=2)

    def forward(self, x):
        return self.mha(x, x, x)


class OrdinaryModelWithRecurrentInName(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 1, 1)

    def forward(self, x):
        quantize_agnostic = x[:2]
        return self.conv(quantize_agnostic)


class ShiftScaleParametrized(torch.nn.Module):
    NUM_CHANNELS = 3
    INPUT_SIZES = [1, NUM_CHANNELS, 2, 2]

    def __init__(self, is_single_input: bool, use_normalize: bool):
        super().__init__()
        self.conv = create_conv(self.NUM_CHANNELS, 1, 1)
        self.is_single_input = is_single_input
        self.use_normalize = use_normalize

    @classmethod
    def get_name(cls, is_single_input: bool, use_normalize: bool):
        suffix_1 = "single" if is_single_input else "multi"
        suffix_2 = "__normalize" if use_normalize else ""
        return f"ShiftScale{suffix_2}__{suffix_1}_input_branch"

    def forward(self, x):
        values = [1] * self.NUM_CHANNELS
        if self.use_normalize:
            pre_proc = normalize(x, values, values, inplace=False)
        else:
            vector = torch.Tensor(values).unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
            pre_proc = (x - vector) / vector

        output = self.conv(pre_proc)
        if self.is_single_input:
            return output
        return output, self.conv(x)


class ModelForGraphBuildingTest(torch.nn.Module):
    IN_CHANNELS = 3
    OUT_CHANNELS = 10
    CONV1_OUT_CHANNELS = 15
    CONV2_IN_CHANNELS = CONV1_OUT_CHANNELS + IN_CHANNELS
    MAXPOOL_SIZE = 2
    INPUT_SHAPES = [(1, 3, 224, 224), (2, 3, 224, 224), (1, 3, 500, 500)]

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(self.IN_CHANNELS, self.CONV1_OUT_CHANNELS, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(15)
        self.relu1 = nn.ReLU()
        self.convt1 = nn.ConvTranspose2d(self.CONV1_OUT_CHANNELS, self.IN_CHANNELS, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(self.CONV2_IN_CHANNELS, self.OUT_CHANNELS, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x_prev = x
        x = F.max_pool2d(x, self.MAXPOOL_SIZE)
        x = self.convt1(x)
        x = torch.cat([x, x_prev], 1)
        x = self.conv2(x)
        return x

    @staticmethod
    def simple_wrap_fn(args, kwargs):
        arglist = list(args)
        arglist[0] = nncf_model_input(arglist[0])
        args = tuple(arglist)
        return args, kwargs

    @classmethod
    def simple_user_dummy_forward(cls, model):
        mock_tensor = torch.zeros(cls.INPUT_SHAPES[0])
        args = (mock_tensor,)
        kwargs = {}
        args, kwargs = cls.simple_wrap_fn(args, kwargs)
        return wrap_nncf_model_outputs_with_objwalk(model(*args, **kwargs))


class ModelForGraphBuildingTestWithConcat(nn.Module):
    INPUT_SHAPE = (1, 1, 1, 1)

    def forward(self, x):
        outputs = []
        outputs.append(torch.stack([x, x]))
        outputs.append(torch.stack(tensors=[x, x, x]))
        outputs.append(torch.stack([x, x, x, x], dim=3))
        outputs.append(torch.stack(tensors=[x, x, x, x, x], dim=2))
        outputs.append(torch.cat([x, x]))
        outputs.append(torch.cat(tensors=[x, x, x]))
        outputs.append(torch.cat([x, x, x, x], dim=3))
        outputs.append(torch.cat(tensors=[x, x, x, x, x], dim=2))
        return outputs


class ModelForGraphBuildingTestWithReshapeFlattenAndConcat(ModelForGraphBuildingTest):
    def forward(self, x):
        y = super().forward(x)
        size = y.size()
        y = y.view(size + (1, 1))

        y_copy = torch.ones_like(y)
        y = torch.stack([y, y_copy])

        y_copy = torch.ones_like(y)
        y = torch.cat([y, y_copy], -1)

        y = torch.flatten(y)
        _ = y.view(-1)

        y_copy = torch.ones_like(y)
        y = torch.stack([y, y_copy])

        y_copy = torch.ones_like(y)
        y = torch.cat([y, y_copy], -1)
        return y


class ModelWithPermute(nn.Module):
    def forward(self, x: torch.Tensor):
        # x.shape == [1, 10, 20, 10]
        # without kwargs
        x = x.transpose(1, 3)
        x = x.permute(3, 2, 1, 0)
        # with kwargs
        x = x.transpose(1, dim1=3)
        x = x.transpose(dim0=1, dim1=3)
        x = x.permute(dims=[3, 2, 1, 0])
        return x


class ModelForGraphBuildingTestWithSplit(ModelForGraphBuildingTest):
    def __init__(self, input_shape):
        super().__init__()
        self.conv3 = nn.Conv2d(5, 10, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(input_shape[0], 1, kernel_size=1, padding=0)

    def forward(self, x):
        y = super().forward(x)
        y1, y2 = torch.chunk(y, chunks=2, dim=1)

        y1 = self.conv3(y1)
        y2 = self.conv3(y2)
        y = torch.cat([y1, y2], axis=1)

        y_unbinded = torch.unbind(y, dim=1)
        unbinded_processed = list(y_unbinded)
        unbinded_processed[0] = self.conv4(y_unbinded[0])
        y = torch.cat(unbinded_processed, axis=0)
        return y


class ConvolutionWithNotTensorBiasModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_w = nn.Parameter(torch.ones((1, 1, 1, 1)))

    def forward(self, x):
        w = self._conv_w + 10
        return nn.functional.conv2d(x, w)


class ConvolutionWithSeveralOutputs(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x, x + 2


class ConvolutionWithAllConstantInputsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_w = nn.Parameter(torch.ones((1, 1, 1, 1)))
        self._conv_i = nn.Parameter(torch.ones((1, 1, 1, 1)))

    def forward(self, x):
        w = self._conv_w + 10
        return x + nn.functional.conv2d(self._conv_i, w)


class ConvolutionWithMinModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._conv_w = nn.Parameter(torch.ones((1, 1, 1, 1)))

    def forward(self, x):
        w = self._conv_w + 10
        t = nn.functional.conv2d(x, w)
        return torch.minimum(t, torch.ones_like(t))


class MultiBranchesConnectedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv_b = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv_c = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.bias = torch.tensor([1])

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(a)
        a += self.bias
        b += self.bias
        y = a + b
        return self.conv_c(y) + self.bias


class MultiBranchesConnectedModelWithConcat(torch.nn.Module):
    INPUT_SIZE = (1, 3, 3, 3)

    def __init__(self):
        super().__init__()
        self.conv_a = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv_b = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1)
        self.conv_c = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=1)
        self.const = nn.Parameter(torch.ones(self.INPUT_SIZE))
        self.bias = torch.tensor([1])

    def forward(self, x):
        a = self.conv_a(x)
        b = self.conv_b(a)
        a += self.bias
        b += self.bias
        y = torch.cat([a, b, self.const], dim=1)
        return self.conv_c(y) + self.bias


class LinearPTQParamsTestModel(nn.Module):
    INPUT_SIZE = None

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(3, 3, 1)
        self.bn1 = create_bn(3)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv2 = create_conv(3, 1, 1)
        self.bn2 = create_bn(1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.avg_pool(x)
        x = self.relu(self.conv2(x))
        x = self.bn2(x)
        return x


class ConstantFoldingTestModel(nn.Module):
    INPUT_SIZE = (1, 3, 3, 3)

    def __init__(self):
        super().__init__()
        self.linear_act = nn.Linear(3, 3)
        self.linear_act.weight.data = 2 * torch.ones((3, 3))

        self.linear_w = nn.Linear(3, 3)
        self.linear_w.weight.data = 3 * torch.ones((3, 3))

        self.param = nn.Parameter(4 * torch.ones((3, 3)))

    def forward(self, x, dummy_disconnected_input):
        y = self.linear_w(self.param)
        # Inplace relu to check
        # that inplace operations are
        # removed as well
        y = torch.relu_(y)
        y += 10
        x = self.linear_act(x)
        return x + y


class ShortTransformer(torch.nn.Module):
    def __init__(self, in_features, num_embeddings, share_weights=False):
        super().__init__()
        self.wte = torch.nn.Embedding(num_embeddings, in_features)
        self.linear = torch.nn.Linear(in_features, in_features)
        self.lm_head = torch.nn.Linear(in_features, num_embeddings)

        if share_weights:
            self.lm_head.weight = self.wte.weight

    def forward(self, input_ids):
        x = self.wte(input_ids)
        x = self.linear(x)
        res = self.lm_head(x)
        return res


class YOLO11N_SDPABlock(torch.nn.Module):
    INPUT_SIZE = (1, 2, 4)

    def __init__(self):
        super().__init__()
        self.kqv = nn.Linear(4, 12, bias=False)
        self.fc = nn.Linear

    def forward(self, x):
        x = self.kqv(x)
        k = x[:, :, :4]
        q = x[:, :, 4:8]
        v = x[:, :, 8:]
        kq = torch.matmul(k, torch.transpose(q, 1, 2))
        kq /= 2**-2
        kq = torch.softmax(kq, -1)
        return torch.matmul(torch.transpose(kq, 1, 2), v)
