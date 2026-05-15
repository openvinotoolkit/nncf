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

from nncf.torch.function_hook.pruning.magnitude.modules import UnstructuredPruningMask
from nncf.torch.function_hook.pruning.magnitude.modules import apply_magnitude_binary_mask


def test_apply_magnitude_sparsity_binary_mask():
    input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.9])
    mask = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    expected_output = torch.tensor([0.0, 0.5, 0.0, 0.9])

    output = apply_magnitude_binary_mask(input_tensor, mask)
    assert torch.equal(output, expected_output)


def test_unstructured_prune_binary_mask_forward():
    shape = (4,)
    prune_mask = UnstructuredPruningMask(shape)
    assert prune_mask.binary_mask.shape == shape
    assert prune_mask.binary_mask.dtype == torch.bool

    prune_mask.binary_mask = torch.tensor([0, 1, 0, 1], dtype=torch.bool)
    input_tensor = torch.tensor([0.1, 0.5, 0.3, 0.9])
    expected_output = torch.tensor([0.0, 0.5, 0.0, 0.9])

    output = prune_mask(input_tensor)
    assert torch.equal(output, expected_output)


def test_unstructured_prune_binary_mask_get_config():
    shape = (4,)
    prune_mask = UnstructuredPruningMask(shape)
    config = prune_mask.get_config()
    assert config["shape"] == shape


def test_unstructured_prune_binary_mask_from_config():
    shape = (4,)
    prune_mask = UnstructuredPruningMask.from_config({"shape": shape})
    assert prune_mask.binary_mask.shape == shape
