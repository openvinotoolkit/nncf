# Copyright (c) 2024 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pytest
import torch
from torch import nn

from nncf.experimental.torch2.function_hook.model_manager import get_const_data
from nncf.experimental.torch2.function_hook.model_manager import get_module_by_name
from nncf.experimental.torch2.function_hook.model_manager import set_const_data
from nncf.experimental.torch2.function_hook.model_manager import split_const_name


@pytest.mark.parametrize(
    "const_name, ref",
    (
        ("conv.weight", ("conv", "weight")),
        ("module.head.conv.bias", ("module.head.conv", "bias")),
        ("param", ("", "param")),
    ),
)
def test_split_const_name(const_name, ref):
    assert split_const_name(const_name) == ref


class ModelToGetModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(1)
        self.seq = nn.Sequential(nn.Identity(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        x = self.seq(x)
        return x


def test_get_module_by_name():
    model = ModelToGetModule()
    assert get_module_by_name("", model) is model
    assert get_module_by_name("bn", model) is model.bn
    assert get_module_by_name("seq.0", model) is model.seq[0]
    assert get_module_by_name("seq.1", model) is model.seq[1]


class ModelGetSetConst(nn.Module):
    param: torch.nn.Parameter
    buffer: torch.Tensor

    def __init__(self):
        super().__init__()
        self.register_parameter("param", nn.Parameter(torch.tensor([1.0])))
        self.register_buffer("buffer", torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.param + self.buffer


def test_get_const_data():
    model = ModelGetSetConst()

    data = get_const_data(model, "param")
    assert isinstance(data, type(model.param.data))
    assert data == model.param.data

    data = get_const_data(model, "buffer")
    assert isinstance(data, type(model.buffer))
    assert data == model.buffer

    with pytest.raises(AttributeError):
        get_const_data(model, "not_exist")


def test_set_const_data():
    model = ModelGetSetConst()

    set_const_data(model, "param", torch.tensor([100.0]))
    assert isinstance(model.param, torch.nn.Parameter)
    assert model.param.data == torch.tensor([100.0])
    assert list(model.parameters())[0].data == torch.tensor([100.0])

    set_const_data(model, "buffer", torch.tensor([200.0]))
    assert isinstance(model.buffer, torch.Tensor) and not isinstance(model.buffer, torch.nn.Parameter)
    assert model.buffer == torch.tensor([200.0])
    assert list(model.buffers())[0] == torch.tensor([200.0])

    with pytest.raises(AttributeError):
        set_const_data(model, "not_exist", None)
