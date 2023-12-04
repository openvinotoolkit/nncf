# Copyright (c) 2023 Intel Corporation
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

from nncf import CompressWeightsMode
from nncf.quantization import compress_weights


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


def test_compress_weights():
    model = ShortTransformer(5, 10)

    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            n_target_modules += 1
            if module.weight.dtype in [torch.uint8, torch.int8]:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules


def test_compress_shared_weights():
    model = ShortTransformer(5, 10, share_weights=True)

    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    n_target_modules = 0

    for _, module in compressed_model.named_children():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            n_target_modules += 1
            if module.weight.dtype in [torch.uint8, torch.int8]:
                n_compressed_weights += 1

    assert n_compressed_weights == n_target_modules

    assert len(compressed_model.wte.pre_ops) > 0

    assert len(compressed_model.wte.pre_ops) == len(compressed_model.lm_head.pre_ops)

    for key, val in compressed_model.wte.pre_ops.items():
        assert compressed_model.lm_head.get_pre_op(key) is val


def test_raise_error_with_int8_and_non_default_ratio(mocker):
    with pytest.raises(AttributeError):
        compress_weights(mocker.Mock(), mode=CompressWeightsMode.INT8, ratio=0.5)


def test_raise_error_with_int8_and_non_default_group_size(mocker):
    with pytest.raises(AttributeError):
        compress_weights(mocker.Mock(), mode=CompressWeightsMode.INT8, group_size=64)


@pytest.mark.parametrize("mode", [CompressWeightsMode.NF4, CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT4_SYM])
def test_raise_error_with_not_int8(mode):
    with pytest.raises(AttributeError):
        dummy_torch_model = torch.nn.Module()
        compress_weights(dummy_torch_model, mode=mode)
