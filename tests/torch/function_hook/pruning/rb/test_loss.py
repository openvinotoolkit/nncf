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

import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.rb.algo import get_pruned_modules
from nncf.torch.function_hook.pruning.rb.losses import RBLoss
from tests.torch.function_hook.pruning.helpers import ConvModel


def test_rb_loss():
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, examples_inputs=example_inputs)
    rb_loss = RBLoss(pruned_model, 0.9)
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=0.1)
    pruned_model.train()
    begin = get_pruned_modules(pruned_model)["conv.weight"].mask.detach().clone()
    for _ in range(10):
        pruned_model(example_inputs)
        optimizer.zero_grad()
        loss = rb_loss()
        loss.backward()
        optimizer.step()
    end = get_pruned_modules(pruned_model)["conv.weight"].mask.detach().clone()

    assert not torch.allclose(begin, end)
