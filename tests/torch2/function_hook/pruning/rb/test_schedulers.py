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

from pytest_mock import MockerFixture

import nncf
from nncf.parameters import PruneMode
from nncf.torch.function_hook.prune.magnitude.schedulers import ExponentialMagnitudePruningScheduler
from nncf.torch.function_hook.prune.magnitude.schedulers import MultiStepMagnitudePruningScheduler
from tests.torch2.function_hook.pruning.helpers import ConvModel


def test_multi_step_scheduler(mocker: MockerFixture):
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()
    model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs)
    steps = {0: 0.1, 2: 0.5, 4: 0.9}
    scheduler = MultiStepMagnitudePruningScheduler(model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, steps=steps)

    ratio_list = [scheduler.current_ratio]
    for _ in range(6):
        scheduler.step()
        ratio_list.append(scheduler.current_ratio)

    assert ratio_list == [0.1, 0.1, 0.1, 0.5, 0.5, 0.9, 0.9]
    scheduler.step(0)
    assert scheduler.current_ratio == 0.1


def test_exponential_scheduler(mocker: MockerFixture):
    model = ConvModel()
    example_inputs = ConvModel.get_example_inputs()
    model = nncf.prune(model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_LOCAL, ratio=0.5, examples_inputs=example_inputs)

    scheduler = ExponentialMagnitudePruningScheduler(
        model, mode=PruneMode.UNSTRUCTURED_MAGNITUDE_GLOBAL, initial_ratio=0.1, target_ratio=0.9, target_epoch=4
    )
    ratio_list = [scheduler.current_ratio]
    for _ in range(6):
        scheduler.step()
        ratio_list.append(round(scheduler.current_ratio, 3))

    assert ratio_list == [0.1, 0.1, 0.48, 0.7, 0.827, 0.9, 0.9]
    scheduler.step(0)
    assert scheduler.current_ratio == 0.1
