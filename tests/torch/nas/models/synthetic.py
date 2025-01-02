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
import math
from enum import Enum
from enum import auto

import torch
from torch import Tensor
from torch import nn

from nncf.torch.utils import get_model_device
from tests.torch.helpers import create_conv
from tests.torch.nas.helpers import do_conv2d
from tests.torch.nas.helpers import ref_kernel_transform


class TwoConvModel(nn.Module):
    INPUT_SIZE = [1, 1, 10, 10]

    def __init__(self, in_channels=1, out_channels=3, kernel_size=5, weight_init=1, bias_init=0):
        super().__init__()
        self.conv1 = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init)
        self.last_conv = create_conv(out_channels, 3, 1)

    def forward(self, x):
        return self.last_conv(self.conv1(x))


class ThreeConvModelMode(Enum):
    ORIGINAL = auto()
    SUPERNET = auto()
    KERNEL_STAGE = auto()
    WIDTH_STAGE = auto()
    DEPTH_STAGE = auto()


class ThreeConvModel(nn.Module):
    #
    #         / ---- \
    # conv1 ->        add -> last_conv
    #         \      /
    #       conv_to_skip
    #
    INPUT_SIZE = [1, 1, 10, 10]

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 5, bias=False, padding=2)
        self.conv_to_skip = create_conv(3, 3, 1, bias=False)
        self.last_conv = create_conv(3, 1, 1)
        self.mode = ThreeConvModelMode.SUPERNET
        self._forward_fn_per_mode = {
            ThreeConvModelMode.SUPERNET: self.supernet_forward,
            ThreeConvModelMode.WIDTH_STAGE: self.forward_min_subnet_on_width_stage,
            ThreeConvModelMode.KERNEL_STAGE: self.forward_min_subnet_on_kernel_stage,
            ThreeConvModelMode.DEPTH_STAGE: self.forward_min_subnet_on_depth_stage,
        }
        self._transition_matrix = nn.Parameter(torch.eye(3**2))

    def assert_weights_equal(self, model: "ThreeConvModel"):
        params = dict(model.named_parameters())
        for name, ref_param in self.named_parameters():
            if name.endswith("transition_matrix"):
                continue
            param = params[name]
            assert torch.equal(ref_param, param)

    def assert_transition_matrix_equals(self, matrix_to_cmp: Tensor):
        assert torch.equal(self._transition_matrix, matrix_to_cmp)

    def supernet_forward(self, x):
        o1 = self.conv1(x)
        o2 = self.conv_to_skip(o1)
        o3 = o1 + o2
        return self.last_conv(o3)

    def forward_min_subnet_on_kernel_stage(self, x):
        ref_weights = ref_kernel_transform(self.conv1.weight, transition_matrix=self._transition_matrix)
        o1 = do_conv2d(self.conv1, x, padding=1, weight=ref_weights)
        o2 = self.conv_to_skip(o1)
        o3 = o1 + o2
        return self.last_conv(o3)

    def forward_min_subnet_on_depth_stage(self, x):
        ref_weights = ref_kernel_transform(self.conv1.weight, transition_matrix=self._transition_matrix)
        o1 = do_conv2d(self.conv1, x, padding=1, weight=ref_weights)
        o3 = o1 + o1
        return self.last_conv(o3)

    def forward_min_subnet_on_width_stage(self, x):
        # NOTE: though the order of kernel/width operations shouldn't lead to a different result mathematically,
        # there may be a minor floating-point error in the 6th sign
        ref_weights = self.conv1.weight[:1, :, :, :]
        ref_weights = ref_kernel_transform(ref_weights, transition_matrix=self._transition_matrix)
        o1 = do_conv2d(self.conv1, x, padding=1, weight=ref_weights)
        o3 = o1 + o1
        ref_weights_last = self.last_conv.weight[:, :1, :, :]
        ref_bias_last = self.last_conv.bias[:1]
        return do_conv2d(self.last_conv, o3, weight=ref_weights_last, bias=ref_bias_last)

    def forward(self, x):
        fn = self._forward_fn_per_mode[self.mode]
        return fn(x)


class TwoSequentialConvBNTestModel(nn.Module):
    #
    # conv1 -> bn1 -> conv2 -> bn2 -> last_conv
    #
    INPUT_SIZE = [1, 1, 1, 1]
    w1_max = 3
    w2_max = 2
    w2_min = 1
    IMPORTANCE = {
        "TwoSequentialConvBNTestModel/Sequential[all_layers]/NNCFConv2d[0]/conv2d_0": torch.Tensor(
            [[[[0.7]]], [[[0.9]]], [[[0.8]]]]
        ),
        "TwoSequentialConvBNTestModel/Sequential[all_layers]/NNCFConv2d[3]/conv2d_0": torch.Tensor(
            [[[[0.0]], [[0.8]], [[1.0]]], [[[0.8]], [[0.7]], [[1.0]]]]
        ),
    }

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 1)
        self.bn1 = nn.BatchNorm2d(3)
        bias = torch.Tensor([self.w1_max, 1, 2])
        weights = bias.reshape(3, 1, 1, 1)
        self.set_params(bias, weights, self.conv1, self.bn1)

        self.conv2 = create_conv(3, 2, 1)
        self.bn2 = nn.BatchNorm2d(2)

        weight = torch.Tensor([[self.w2_min, 2, 0], [self.w2_max, 3, 0]]).reshape(2, 3, 1, 1)
        bias = torch.Tensor([1, self.w2_max])
        self.set_params(bias, weight, self.conv2, self.bn2)

        self.last_conv = create_conv(2, 1, 1)
        self.all_layers = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.conv2, self.bn2, nn.ReLU(), self.last_conv
        )

    @staticmethod
    def set_params(bias, weight, conv, bn):
        conv.weight.data = weight
        list_params = TwoSequentialConvBNTestModel.get_bias_like_params(bn, conv)
        for param in list_params:
            param.data = bias

    @staticmethod
    def compare_params(bias, weight, conv, bn):
        device = conv.weight.device
        weight.to(device)
        assert torch.equal(conv.weight, weight)
        list_params = TwoSequentialConvBNTestModel.get_bias_like_params(bn, conv)
        for param in list_params:
            param.to(bias.device)
            assert torch.equal(param, bias)

    @staticmethod
    def get_bias_like_params(bn, conv):
        list_params = [conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var]
        return list_params

    def check_reorg(self):
        device = next(self.parameters()).device
        ref_bias_1 = torch.Tensor([self.w1_max, 2, 1]).to(device)
        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1).to(device)
        ref_bias_2 = torch.Tensor([self.w2_max, 1]).to(device)
        ref_weights_2 = torch.Tensor([[self.w2_max, 0, 3], [self.w2_min, 0, 2]]).reshape(2, 3, 1, 1).to(device)
        TwoSequentialConvBNTestModel.compare_params(ref_bias_1, ref_weights_1, self.conv1, self.bn1)
        TwoSequentialConvBNTestModel.compare_params(ref_bias_2, ref_weights_2, self.conv2, self.bn2)

        last_bias = self.last_conv.bias
        assert torch.equal(self.last_conv.weight, torch.Tensor([2, 2]).reshape(1, 2, 1, 1).to(device))
        assert torch.equal(last_bias, torch.zeros_like(last_bias).to(device))

    def check_custom_external_reorg(self):
        device = next(self.parameters()).device
        ref_bias_1 = torch.Tensor([1, 2, self.w1_max]).to(device)
        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1).to(device)
        ref_bias_2 = torch.Tensor([self.w2_max, 1]).to(device)
        ref_weights_2 = torch.Tensor([[3, 0, self.w2_max], [2, 0, self.w2_min]]).reshape(2, 3, 1, 1).to(device)
        TwoSequentialConvBNTestModel.compare_params(ref_bias_1, ref_weights_1, self.conv1, self.bn1)
        TwoSequentialConvBNTestModel.compare_params(ref_bias_2, ref_weights_2, self.conv2, self.bn2)

        last_bias = self.last_conv.bias
        assert torch.equal(self.last_conv.weight, torch.Tensor([2, 2]).reshape(1, 2, 1, 1).to(device))
        assert torch.equal(last_bias, torch.zeros_like(last_bias).to(device))

    def get_minimal_subnet_output(self, input_):
        relu1 = self._get_relu1_output(input_)
        return self._get_model_output(relu1, self.w2_max)

    def _get_model_output(self, relu1, value):
        conv2_output = relu1 * value + value
        bn2_output = ((conv2_output - value) / math.sqrt(self.bn2.eps + value)) * value + value
        relu2 = nn.ReLU()(bn2_output)

        ref_weights = self.last_conv.weight[:, :1, :, :]
        ref_bias = self.last_conv.bias[:1]
        return do_conv2d(self.last_conv, relu2, weight=ref_weights, bias=ref_bias)

    def _get_relu1_output(self, input_):
        value = self.w1_max
        conv1_output = input_ * value + value
        bn1_output = ((conv1_output - value) / math.sqrt(self.bn1.eps + value)) * value + value
        relu1 = nn.ReLU()(bn1_output)
        return relu1

    def get_minimal_subnet_output_without_reorg(self, input_):
        relu1 = self._get_relu1_output(input_)
        return self._get_model_output(relu1, self.w2_min)

    def forward(self, x):
        return self.all_layers(x)


class TwoConvAddConvTestModel(nn.Module):
    #
    # conv1 - \
    #         add -> last_conv
    # conv2 - /
    #
    INPUT_SIZE = [1, 1, 1, 1]
    V13 = 2
    V23 = 4
    V11 = 3
    V21 = 1
    IMPORTANCE = {
        "TwoConvAddConvTestModel/NNCFConv2d[conv1]/conv2d_0": torch.Tensor([[[[0.7]]], [[[0.9]]], [[[0.8]]]]),
        "TwoConvAddConvTestModel/NNCFConv2d[conv2]/conv2d_0": torch.Tensor([[[[0.1]]], [[[0.8]]], [[[0.7]]]]),
    }

    def __init__(self):
        super().__init__()
        self.conv1 = create_conv(1, 3, 1)
        self.conv2 = create_conv(1, 3, 1)
        self.last_conv = create_conv(3, 1, 1)
        self._init_params(self.conv1, torch.Tensor([self.V11, 1, self.V13]))
        self._init_params(self.conv2, torch.Tensor([self.V21, 2, self.V23]))

    @staticmethod
    def _init_params(conv, data):
        conv.bias.data = data
        weight = data.reshape(3, 1, 1, 1)
        conv.weight.data = weight

    @staticmethod
    def get_bias_like_params(bn, conv):
        list_params = [conv.bias, bn.weight, bn.bias, bn.running_mean, bn.running_var]
        return list_params

    def check_reorg(self):
        device = get_model_device(self)
        ref_bias_1 = torch.Tensor([self.V13, self.V11, 1]).to(device)
        ref_bias_2 = torch.Tensor([self.V23, self.V21, 2]).to(device)

        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1).to(device)
        ref_weights_2 = ref_bias_2.reshape(3, 1, 1, 1).to(device)

        assert torch.equal(self.conv1.weight, ref_weights_1)
        assert torch.equal(self.conv1.bias, ref_bias_1)
        assert torch.equal(self.conv2.weight, ref_weights_2)
        assert torch.equal(self.conv2.bias, ref_bias_2)
        last_bias = self.last_conv.bias
        assert torch.equal(self.last_conv.weight, torch.Tensor([2, 2, 2]).reshape(1, 3, 1, 1).to(device))
        assert torch.equal(last_bias, torch.zeros_like(last_bias).to(device))

    def check_custom_external_reorg(self):
        device = get_model_device(self)
        ref_bias_1 = torch.Tensor([1, self.V13, self.V11]).to(device)
        ref_bias_2 = torch.Tensor([2, self.V23, self.V21]).to(device)

        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1).to(device)
        ref_weights_2 = ref_bias_2.reshape(3, 1, 1, 1).to(device)

        assert torch.equal(self.conv1.weight, ref_weights_1)
        assert torch.equal(self.conv1.bias, ref_bias_1)
        assert torch.equal(self.conv2.weight, ref_weights_2)
        assert torch.equal(self.conv2.bias, ref_bias_2)
        last_bias = self.last_conv.bias
        assert torch.equal(self.last_conv.weight, torch.Tensor([2, 2, 2]).reshape(1, 3, 1, 1).to(device))
        assert torch.equal(last_bias, torch.zeros_like(last_bias).to(device))

    def get_minimal_subnet_output(self, x):
        o = (self.V13 * x + self.V13) + (self.V23 * x + self.V23)
        ref_weights = self.last_conv.weight[:, :1, :, :]
        ref_bias = self.last_conv.bias[:1]
        return do_conv2d(self.last_conv, o, weight=ref_weights, bias=ref_bias)

    def get_minimal_subnet_output_without_reorg(self, x):
        o = (self.V11 * x + self.V11) + (self.V21 * x + self.V21)
        ref_weights = self.last_conv.weight[:, :1, :, :]
        ref_bias = self.last_conv.bias[:1]
        return do_conv2d(self.last_conv, o, weight=ref_weights, bias=ref_bias)

    def forward(self, x):
        o = self.conv1(x) + self.conv2(x)
        return self.last_conv(o)


class ConvTwoFcTestModel(nn.Module):
    #
    # conv -> fc1 -> fc2
    #
    INPUT_SIZE = [1, 1, 1, 1]
    V11 = 2
    V13 = 4
    IMPORTANCE = {
        "ConvTwoFcTestModel/NNCFConv2d[conv]/conv2d_0": torch.Tensor([[[[0.9]]], [[[0.1]]], [[[0.2]]]]),
        "ConvTwoFcTestModel/NNCFLinear[fc1]/linear_0": torch.Tensor([[0.7, 0.9, 0.8], [0.6, 0.4, 0.5]]),
    }

    def __init__(self):
        super().__init__()
        self.conv = create_conv(1, 3, 1)
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 3)

        self._init_params_conv(self.conv, torch.Tensor([self.V11, 1, self.V13]))
        self._init_params_fc(self.fc1, torch.Tensor([[3, 1, 2], [4, 6, 5]]))
        self._init_params_fc(self.fc2, torch.Tensor([[2, 1], [3, 4], [6, 5]]))

    @staticmethod
    def _init_params_conv(conv, data):
        conv.bias.data = data
        weight = data.reshape(3, 1, 1, 1)
        conv.weight.data = weight

    @staticmethod
    def _init_params_fc(fc, data):
        fc.weight.data = data
        fc.bias.data = data[:, 0]

    def check_reorg(self):
        device = get_model_device(self)
        ref_bias_1 = torch.Tensor([self.V13, self.V11, 1]).to(device)
        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1).to(device)

        fc_weights_1 = torch.Tensor([[5, 4, 6], [2, 3, 1]]).to(device)
        fc_bias_1 = torch.Tensor([4, 3]).to(device)
        fc_weights_2 = torch.Tensor([[1, 2], [4, 3], [5, 6]]).to(device)  # last layer don't change the output order

        assert torch.equal(self.conv.weight, ref_weights_1)
        assert torch.equal(self.conv.bias, ref_bias_1)
        assert torch.equal(self.fc1.weight, fc_weights_1)
        assert torch.equal(self.fc1.bias, fc_bias_1)
        assert torch.equal(self.fc2.weight, fc_weights_2)

    def check_custom_external_reorg(self):
        device = get_model_device(self)
        ref_bias_1 = torch.Tensor([self.V11, self.V13, 1]).to(device)
        ref_weights_1 = ref_bias_1.reshape(3, 1, 1, 1).to(device)

        fc_weights_1 = torch.Tensor([[3, 2, 1], [4, 5, 6]]).to(device)
        fc_bias_1 = torch.Tensor([3, 4]).to(device)
        fc_weights_2 = torch.Tensor([[2, 1], [3, 4], [6, 5]]).to(device)  # last layer don't change the output order

        assert torch.equal(self.conv.weight, ref_weights_1)
        assert torch.equal(self.conv.bias, ref_bias_1)
        assert torch.equal(self.fc1.weight, fc_weights_1)
        assert torch.equal(self.fc1.bias, fc_bias_1)
        assert torch.equal(self.fc2.weight, fc_weights_2)

    def get_minimal_subnet_output(self, x):
        device = get_model_device(self)
        fc_weight_1 = torch.Tensor([[5]]).to(device)
        fc_weight_2 = torch.Tensor([[1], [4], [5]]).to(device)
        fc_bias_2 = torch.Tensor([2, 3, 6]).reshape(1, 1, 1, 3).to(device)

        x = self.V13 * x + self.V13  # conv1
        x = x * fc_weight_1.t() + 4  # fc1
        x = x * fc_weight_2.t() + fc_bias_2  # fc2
        return x

    def forward(self, x):
        x = self.conv(x)
        x = x.view(1, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TwoSequentialFcLNTestModel(nn.Module):
    #
    # fc1 -> ln1 -> fc2 -> ln2
    #
    INPUT_SIZE = [1, 1]
    IMPORTANCE = {
        "TwoSequentialFcLNTestModel/NNCFLinear[fc1]/linear_0": torch.Tensor([[0.9], [0.1]]),
    }

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 2)
        self.fc2 = nn.Linear(2, 3)
        self.ln1 = nn.LayerNorm(2)
        self.ln2 = nn.LayerNorm(3)

        ConvTwoFcTestModel._init_params_fc(self.fc1, torch.Tensor([[3], [4]]))
        ConvTwoFcTestModel._init_params_fc(self.fc2, torch.Tensor([[2, 1], [3, 4], [6, 5]]))
        self._init_params_ln(self.ln1)
        self._init_params_ln(self.ln2)

    @staticmethod
    def _init_params_ln(ln):
        ln.weight.data = torch.arange(len(ln.weight.data)).float()
        ln.bias.data = torch.arange(len(ln.bias.data)).float()

    def check_reorg(self):
        device = get_model_device(self)
        ref_fc_weights_1 = torch.Tensor([[4], [3]]).to(device)
        ref_fc_bias_1 = torch.Tensor([4, 3]).to(device)
        ref_ln_weights_1 = torch.Tensor([1, 0]).to(device)
        ref_ln_bias_1 = torch.Tensor([1, 0]).to(device)
        ref_fc_weights_2 = torch.Tensor([[1, 2], [4, 3], [5, 6]]).to(device)
        assert torch.equal(self.fc1.weight, ref_fc_weights_1)
        assert torch.equal(self.fc1.bias, ref_fc_bias_1)
        assert torch.equal(self.fc2.weight, ref_fc_weights_2)
        assert torch.equal(self.ln1.weight, ref_ln_weights_1)
        assert torch.equal(self.ln1.bias, ref_ln_bias_1)

    def check_custom_external_reorg(self):
        device = get_model_device(self)
        ref_fc_weights_1 = torch.Tensor([[3], [4]]).to(device)
        ref_fc_bias_1 = torch.Tensor([3, 4]).to(device)
        ref_ln_weights_1 = torch.Tensor([0, 1]).to(device)
        ref_ln_bias_1 = torch.Tensor([0, 1]).to(device)
        ref_fc_weights_2 = torch.Tensor([[2, 1], [3, 4], [6, 5]]).to(device)
        assert torch.equal(self.fc1.weight, ref_fc_weights_1)
        assert torch.equal(self.fc1.bias, ref_fc_bias_1)
        assert torch.equal(self.fc2.weight, ref_fc_weights_2)
        assert torch.equal(self.ln1.weight, ref_ln_weights_1)
        assert torch.equal(self.ln1.bias, ref_ln_bias_1)

    def get_minimal_subnet_output(self, x):
        device = get_model_device(self)
        fc_weight_1 = torch.Tensor([[4]]).to(device)
        fc_weight_2 = torch.Tensor([[1], [4], [5]]).to(device)
        fc_bias_2 = torch.Tensor([2, 3, 6]).reshape(1, 1, 3).to(device)
        ln_weight_2 = torch.Tensor([0, 1, 2]).to(device)
        ln_bias_2 = torch.Tensor([0, 1, 2]).to(device)

        x = x * fc_weight_1.t() + 4  # fc1
        x = (x - 8) / math.sqrt(self.ln1.eps + 8) * 1 + 1  # ln1
        x = x * fc_weight_2.t() + fc_bias_2  # fc2
        x = (x - 7) / math.sqrt(self.ln2.eps + 32 / 3) * ln_weight_2 + ln_bias_2  # ln2
        return x

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.fc2(x)
        x = self.ln2(x)
        return x


class TwoConvMeanModel(nn.Module):
    INPUT_SIZE = [1, 1, 10, 10]

    def __init__(self, in_channels=1, out_channels=3, kernel_size=5, weight_init=1, bias_init=0):
        super().__init__()
        self.conv1 = create_conv(in_channels, out_channels, kernel_size, weight_init, bias_init)
        self.last_conv = create_conv(out_channels, 3, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = x.mean(2, keepdim=True).mean(3, keepdim=True)
        x = self.last_conv(x)
        return x
