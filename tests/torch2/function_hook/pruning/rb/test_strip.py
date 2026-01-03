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


import torch

import nncf
from nncf import PruneMode
from nncf.torch.function_hook.pruning.rb.modules import RBPruningMask
from nncf.torch.function_hook.wrapper import get_hook_storage
from tests.torch2.function_hook.pruning.helpers import ConvModel


def test_strip():
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()
    pruned_model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, examples_inputs=example_inputs)
    pruned_model.eval()

    hook_storage = get_hook_storage(pruned_model)
    pruning_module = hook_storage.post_hooks["conv:weight__0"]["0"]

    assert isinstance(pruning_module, RBPruningMask)

    with torch.no_grad():
        # Set mask
        pruning_module.mask[0] *= -1
        pruned_weight = pruning_module(pruned_model.conv.weight)

    striped_model = nncf.strip(pruned_model, strip_format=nncf.StripFormat.IN_PLACE, do_copy=False)
    hook_storage = get_hook_storage(striped_model)

    assert not list(hook_storage.named_hooks())
    assert torch.equal(striped_model.conv.weight, pruned_weight)
    assert torch.count_nonzero(striped_model.conv.weight) == 54
