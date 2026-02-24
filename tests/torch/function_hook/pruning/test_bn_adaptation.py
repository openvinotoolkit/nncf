# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import Optional

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

import nncf
from nncf.torch.function_hook.pruning.batch_norm_adaptation import set_batchnorm_train_only


class ModelBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 3)
        self.bn1 = nn.BatchNorm2d(3)
        self.seq = nn.Sequential(nn.Linear(3, 3))

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = self.seq(x)
        return x


def test_set_batchnorm_train_only():
    model = ModelBN()
    model.train()

    # Set one layer to check restoration
    model.seq[0].eval()

    with set_batchnorm_train_only(model):
        for module in model.modules():
            if isinstance(module, _BatchNorm):
                assert module.training, "BatchNorm layer should be in training mode"
            else:
                assert not module.training, "Non-BatchNorm layer should be in eval mode"

    # After exiting the context manager, all modules should be back to training mode
    for module in model.modules():
        if module is model.seq[0]:
            assert not module.training
        else:
            assert module.training


@pytest.mark.parametrize("num,ref", ((None, 5), (2, 2)))
def test_batch_norm_adaptation(mocker: MockerFixture, num: Optional[int], ref: int):
    model = ModelBN()

    dataloader = torch.utils.data.DataLoader([(torch.randn((3, 3, 3)), 1)] * 10, batch_size=2)

    def transform_fn(batch: tuple[Tensor, int]):
        inputs, _ = batch
        return inputs

    nncf_dataset = nncf.Dataset(dataloader, transform_fn)
    spy_func = mocker.spy(ModelBN, "forward")

    org_state = deepcopy(model.state_dict())
    model = nncf.batch_norm_adaptation(model, calibration_dataset=nncf_dataset, num_iterations=num)

    assert spy_func.call_count == ref
    new_state = model.state_dict()

    for x in org_state:
        if x in ["bn1.running_mean", "bn1.running_var", "bn1.num_batches_tracked"]:
            assert not torch.equal(org_state[x], new_state[x]), f"State {x} did not change"
        else:
            if isinstance(org_state[x], torch.Tensor):
                assert torch.equal(org_state[x], new_state[x]), f"State {x} changed"


@pytest.mark.parametrize("in_type", ["dict", "tuple", "tensor"])
def test_batch_norm_adaptation_inputs(mocker: MockerFixture, in_type: str):
    model = ModelBN()

    dataloader = torch.utils.data.DataLoader([(torch.randn((3, 3, 3)), 1)] * 10, batch_size=2)

    def transform_fn(batch: tuple[Tensor, int]):
        if in_type == "dict":
            return {"x": batch[0]}
        if in_type == "tuple":
            return (batch[0],)
        if in_type == "tensor":
            return batch[0]

    nncf_dataset = nncf.Dataset(dataloader, transform_fn)
    spy_func = mocker.spy(ModelBN, "forward")

    model = nncf.batch_norm_adaptation(model, calibration_dataset=nncf_dataset, num_iterations=2)
    assert spy_func.call_count == 2
