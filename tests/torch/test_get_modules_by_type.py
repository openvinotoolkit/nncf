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
from collections import OrderedDict

import pytest
import torch.nn.functional as F
from torch.nn import AvgPool2d
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential

from nncf.torch.utils import get_all_modules_by_type


class ModelForNameTest(Module):
    def __init__(self, size=1):
        super().__init__()
        self.size = size
        self.conv0 = Conv2d(size, size, size)
        self.conv1 = Conv2d(size, size, size)
        self.bn1 = BatchNorm2d(size)
        self.bn2 = BatchNorm2d(size)
        self.norm10 = BatchNorm2d(size)
        self.norm20 = BatchNorm2d(size)
        self.avgpool = AvgPool2d(size)
        self.layer1 = Sequential(
            OrderedDict(
                [("conv01", self.conv0), ("norm01", self.norm10), ("relu01", ReLU()), ("pool01", MaxPool2d(size))]
            )
        )
        self.layer2 = Sequential(
            OrderedDict(
                [
                    ("layer1", self.layer1),
                    ("conv02", self.conv0),
                    ("relu02", ReLU()),
                    ("norm02", self.norm20),
                    ("pool02", MaxPool2d(size)),
                ]
            )
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = ReLU()(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.layer2(x)
        x = self.avgpool(x)
        return x


def test_get_all_modules_by_type__for_standard_type():
    model = ModelForNameTest()
    act_bn = get_all_modules_by_type(model, "BatchNorm2d")
    act_bn = OrderedDict((str(k), v) for k, v in act_bn.items())
    ref_bn = {
        "ModelForNameTest/BatchNorm2d[bn1]": model.bn1,
        "ModelForNameTest/BatchNorm2d[bn2]": model.bn2,
        "ModelForNameTest/BatchNorm2d[norm10]": model.norm10,
        "ModelForNameTest/BatchNorm2d[norm20]": model.norm20,
    }
    assert act_bn == ref_bn


def test_get_all_modules_by_type__for_multiple_type():
    model = ModelForNameTest()
    act_bn = get_all_modules_by_type(model, ["ReLU", "AvgPool2d"])
    act_bn = OrderedDict((str(k), v) for k, v in act_bn.items())
    ref_bn = [
        "ModelForNameTest/AvgPool2d[avgpool]",
        "ModelForNameTest/Sequential[layer1]/ReLU[relu01]",
        "ModelForNameTest/Sequential[layer2]/ReLU[relu02]",
    ]
    assert list(act_bn.keys()) == ref_bn
    assert isinstance(act_bn, OrderedDict)


def test_get_all_modules_by_type__for_not_exact_type():
    model = ModelForNameTest()
    layer = get_all_modules_by_type(model, "Avg")
    assert not layer


def test_get_all_modules_by_type__for_subtype():
    model = ModelForNameTest()
    layer = get_all_modules_by_type(model, "AvgPool2d_dummy")
    assert not layer


IGNORED_SCOPES = [
    ("single", ["ModelForNameTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]"]),
    (
        "multiple",
        [
            "ModelForNameTest/Sequential[layer2]/Sequential[layer1]/ReLU[relu01]",
            "ModelForNameTest/Sequential[layer2]/Conv2d[conv02]",
        ],
    ),
    ("common", ["ModelForNameTest/Sequential[layer1]"]),
]


@pytest.mark.parametrize("ignored_scopes", [s[1] for s in IGNORED_SCOPES], ids=[s[0] for s in IGNORED_SCOPES])
def test_get_all_modules_by_type__with_ignored_scope(ignored_scopes):
    model = ModelForNameTest()

    model_modules = set()
    for _, module in model.named_modules():
        model_modules.add(module.__class__.__name__)
    model_modules = list(model_modules)

    act_modules = get_all_modules_by_type(model, model_modules, ignored_scopes=ignored_scopes)

    for module_scope, _ in act_modules.items():
        for scope in ignored_scopes:
            assert not str(module_scope).startswith(str(scope))
