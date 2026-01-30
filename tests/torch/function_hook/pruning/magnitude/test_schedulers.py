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
from nncf.torch.function_hook.pruning.magnitude.algo import get_pruned_modules
from nncf.torch.function_hook.pruning.magnitude.schedulers import ExponentialMagnitudePruningScheduler
from nncf.torch.function_hook.pruning.magnitude.schedulers import MultiStepMagnitudePruningScheduler
from tests.torch.function_hook.pruning.helpers import ConvModel


def test_multi_step_scheduler():
    model = ConvModel()
    model.conv.weight.data = torch.arange(0, 81).view(3, 3, 3, 3).float()
    example_inputs = ConvModel.get_example_inputs()
    model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.2, examples_inputs=example_inputs)
    scheduler = MultiStepMagnitudePruningScheduler(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, steps={0: 0.1, 1: 0.8}
    )

    mask = get_pruned_modules(model)["conv.weight"].binary_mask
    r = float(1 - mask.sum() / 81)
    assert r == 0.20987653732299805
    scheduler.step()
    r = float(1 - mask.sum() / 81)
    assert r == 0.1111111044883728
    scheduler.step()
    r = float(1 - mask.sum() / 81)
    assert r == 0.8024691343307495


def test_exponential_scheduler():
    model = ConvModel()
    model.conv.weight.data = torch.arange(0, 81).view(3, 3, 3, 3).float()

    example_inputs = ConvModel.get_example_inputs()
    model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs)

    scheduler = ExponentialMagnitudePruningScheduler(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, initial_ratio=0.1, target_ratio=0.9, target_epoch=4
    )

    mask = get_pruned_modules(model)["conv.weight"].binary_mask
    r = float(1 - mask.sum() / 81)
    assert r == 0.5061728358268738
    scheduler.step()
    r = float(1 - mask.sum() / 81)
    assert r == 0.1111111044883728
    scheduler.step()
    r = float(1 - mask.sum() / 81)
    assert r == 0.48148149251937866
