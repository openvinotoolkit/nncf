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

import torch

from nncf.quantization import compress_weights


class ShortTransformer(torch.nn.Module):
    def __init__(self, in_features, num_embeddings):
        super().__init__()
        self.wte = torch.nn.Embedding(num_embeddings, in_features)
        self.linear = torch.nn.Linear(in_features, in_features)
        self.lm_head = torch.nn.Linear(in_features, num_embeddings)

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
