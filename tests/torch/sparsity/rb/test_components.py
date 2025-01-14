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

import pytest
import torch
from torch import nn

from nncf.common.hook_handle import HookHandle
from nncf.torch.layers import NNCFConv2d
from nncf.torch.layers import NNCFConvTranspose2d
from nncf.torch.layers import NNCFLinear
from nncf.torch.module_operations import UpdateWeight
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight
from nncf.torch.sparsity.rb.loss import SparseLoss


class SingleLayerModel(nn.Module):
    def __init__(self, layer, frozen, size=1):
        super().__init__()
        self.size = size
        self.layer = layer
        if frozen is None:
            sparsifier = RBSparsifyingWeight(weight_shape=size)
        else:
            sparsifier = RBSparsifyingWeight(frozen=frozen, weight_shape=size)
        self.hook_handle: HookHandle = self.layer.register_pre_forward_operation(UpdateWeight(sparsifier))

    @property
    def sparsifier(self):
        return self.layer.get_pre_op(self.hook_handle.hook_id).operand

    def forward(self, x):
        return self.layer(x)


def sparse_model(module, frozen, size=1):
    layer = module(size, size, size)
    return SingleLayerModel(layer, frozen, size)


@pytest.mark.parametrize("module", [NNCFLinear, NNCFConv2d, NNCFConvTranspose2d])
class TestSparseModules:
    def test_create_loss__with_defaults(self, module):
        model = sparse_model(module, None)
        loss = SparseLoss([model.sparsifier])
        assert not loss.disabled
        assert loss.target_sparsity_rate == 0
        assert loss.p == 0.05

    @pytest.mark.parametrize(
        ("mask_value", "ref_loss"),
        ((None, 1), (0, 0), (0.3, 1), (-0.3, 0)),
        ids=("default", "zero", "positive", "negative"),
    )
    def test_can_forward_sparse_module__with_frozen_mask(self, module, mask_value, ref_loss):
        model = sparse_model(module, True)
        sm = model.layer
        sm.weight.data.fill_(1)
        sm.bias.data.fill_(0)
        sw = model.sparsifier
        if mask_value is not None:
            new_mask = torch.zeros_like(sw.mask)
            new_mask.fill_(mask_value)
            sw.mask = new_mask
        input_ = torch.ones([1, 1, 1, 1])
        assert model(input_).item() == ref_loss

    @pytest.mark.parametrize(
        ("frozen", "raising"), ((None, True), (True, True), (False, False)), ids=("default", "frozen", "not_frozen")
    )
    def test_calc_loss(self, module, frozen, raising):
        model = sparse_model(module, frozen)
        sw = model.sparsifier
        assert sw.frozen is (True if frozen is None else frozen)
        loss = SparseLoss([model.sparsifier])
        try:
            assert loss() == 0
        except ZeroDivisionError:
            pytest.fail("Division by zero")
        except AssertionError:
            if not raising:
                pytest.fail("Exception is not expected")

    @pytest.mark.parametrize("frozen", (None, False, True), ids=("default", "sparsify", "frozen"))
    class TestWithSparsify:
        def test_can_freeze_mask(self, module, frozen):
            model = sparse_model(module, frozen)
            sw = model.sparsifier
            if frozen is None:
                frozen = True
            assert sw.frozen is frozen
            assert sw.mask.numel() == 1

        def test_disable_loss(self, module, frozen):
            model = sparse_model(module, frozen)
            sw = model.sparsifier
            assert sw.frozen is (True if frozen is None else frozen)
            loss = SparseLoss([model.sparsifier])
            loss.disable()
            assert sw.frozen

    @pytest.mark.parametrize(
        ("target", "expected_rate"),
        ((None, 0), (0, 1), (0.5, 0.5), (1, 0), (1.5, None), (-0.5, None)),
        ids=("default", "min", "middle", "max", "more_than_max", "less_then_min"),
    )
    def test_get_target_sparsity_rate(self, module, target, expected_rate):
        model = sparse_model(module, None)
        loss = SparseLoss([model.sparsifier])
        if target is not None:
            loss.target = target
        actual_rate = None
        try:
            actual_rate = loss.target_sparsity_rate
            if expected_rate is None:
                pytest.fail("Exception should be raised")
        except IndexError:
            if expected_rate is not None:
                pytest.fail("Exception is not expected")
        if expected_rate is not None:
            assert actual_rate == expected_rate
