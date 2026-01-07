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

import torch

from nncf.torch.function_hook.pruning.rb.modules import RBPruningMask
from nncf.torch.function_hook.pruning.rb.modules import apply_rb_binary_mask


def test_apply_rb_binary_mask():
    input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.9])
    mask = torch.tensor([-1, 1, -1, 1], dtype=torch.float32)
    expected_output = torch.tensor([0.0, 0.5, 0.0, 0.9])

    output = apply_rb_binary_mask(input_tensor, mask, training=False)
    assert torch.allclose(output, expected_output)


def test_rb_prune_binary_mask_forward():
    shape = (4,)
    prune_mask = RBPruningMask(shape).eval()
    assert prune_mask.mask.shape == shape
    assert prune_mask.mask.dtype == torch.float32

    prune_mask.mask.data = torch.tensor([-1, 1, -1, 1], dtype=torch.float32)
    input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.9])
    expected_output = torch.tensor([0.0, 0.5, 0.0, 0.9])

    output = prune_mask(input_tensor)
    assert torch.allclose(output, expected_output)


def test_rb_prune_binary_mask_get_config():
    shape = (4,)
    prune_mask = RBPruningMask(shape)
    config = prune_mask.get_config()
    assert config["shape"] == shape


def test_rb_prune_binary_mask_from_config():
    shape = (4,)
    prune_mask = RBPruningMask.from_config({"shape": shape})
    assert prune_mask.mask.shape == shape
