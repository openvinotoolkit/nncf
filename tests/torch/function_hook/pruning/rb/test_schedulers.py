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


import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.pruning.rb.losses import RBLoss
from nncf.torch.function_hook.pruning.rb.schedulers import ExponentialRBPruningScheduler
from nncf.torch.function_hook.pruning.rb.schedulers import MultiStepRBPruningScheduler
from tests.torch.function_hook.pruning.helpers import ConvModel


def test_multi_step_scheduler():
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()
    model = nncf.prune(
        model, mode=PruneMode.UNSTRUCTURED_REGULARIZATION_BASED, ratio=0.5, examples_inputs=example_inputs
    )
    rb_loss = RBLoss(model, 0.1)
    scheduler = MultiStepRBPruningScheduler(rb_loss, steps={0: 0.1, 2: 0.5})

    assert rb_loss.target_ratio == 0.1
    scheduler.step(0)
    assert rb_loss.target_ratio == 0.1
    scheduler.step(3)
    assert rb_loss.target_ratio == 0.5


def test_exponential_scheduler():
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()
    model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs)
    rb_loss = RBLoss(model, 0.1)

    scheduler = ExponentialRBPruningScheduler(rb_loss, initial_ratio=0.1, target_ratio=0.5, target_epoch=2)
    assert rb_loss.target_ratio == 0.1
    scheduler.step(0)
    assert rb_loss.target_ratio == 0.1
    scheduler.step(3)
    assert rb_loss.target_ratio == 0.5
