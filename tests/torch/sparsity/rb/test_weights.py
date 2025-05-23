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

from nncf.torch.functions import logit
from nncf.torch.sparsity.rb.layers import RBSparsifyingWeight

pytestmark = pytest.mark.legacy
default_mask = logit(torch.ones(1) * 0.99)


def test_can_create_sparse_weight__with_defaults():
    sw = RBSparsifyingWeight(1)
    assert sw.frozen
    assert torch.allclose(default_mask, sw.mask)
    assert not sw.mask.requires_grad
    assert sw.eps == 1e-6


def test_can_freeze_mask():
    sw = RBSparsifyingWeight(1, frozen=False)
    assert not sw.frozen
    assert sw.mask.requires_grad
    sw.frozen = True
    assert sw.frozen
    # NOTE: should be False, but we didn't experiment on real models
    assert sw.mask.requires_grad


@pytest.mark.parametrize("frozen", (True, False), ids=("sparsify", "frozen"))
class TestWithSparsify:
    @pytest.mark.parametrize("is_train", (True, False), ids=("train", "not_train"))
    def test_mask_is_not_updated_on_forward(self, frozen, is_train):
        sw = RBSparsifyingWeight(1, frozen=frozen)
        if is_train:
            sw.train()
        assert torch.allclose(default_mask, sw.mask)
        w = torch.ones(1)
        sw.forward(w)
        assert torch.allclose(default_mask, sw.mask)

    @pytest.mark.parametrize(
        ("mask_value", "ref_loss"),
        ((None, 1), (0, 0), (0.3, 1), (-0.3, 0)),
        ids=("default", "zero", "positive", "negative"),
    )
    def test_loss_value(self, mask_value, ref_loss, frozen):
        sw = RBSparsifyingWeight(1, frozen=frozen)
        if mask_value is not None:
            tensor_to_set = torch.zeros_like(sw.mask)
            tensor_to_set.fill_(mask_value)
            sw.mask = tensor_to_set
        assert sw.loss() == ref_loss
        w = torch.ones(1)
        assert sw.apply_binary_mask(w) == ref_loss
        sw.frozen = True
        assert sw.forward(w) == ref_loss
