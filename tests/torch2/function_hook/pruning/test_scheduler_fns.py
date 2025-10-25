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

import nncf
from nncf.torch.function_hook.pruning.scheduler_fns import exponential_ratio_scheduler
from nncf.torch.function_hook.pruning.scheduler_fns import multi_step_ratio_scheduler


@pytest.mark.parametrize(
    "epoch,expected",
    [
        (0, 0.2),
        (2, 0.2),
        (4, 0.5),
        (5, 0.5),
        (7, 0.3),
        (8, 0.3),
        (10, 0.8),
        (20, 0.8),
    ],
)
def test_multi_step_ratio_scheduler(epoch: int, expected: float):
    steps = {0: 0.2, 4: 0.5, 6: 0.3, 10: 0.8}

    r = multi_step_ratio_scheduler(epoch, steps=steps)
    assert r == expected


@pytest.mark.parametrize(
    "epoch,steps,error_msg",
    [
        (-1, {}, "Epoch"),
        (1, {1: 0.1}, "epoch 0"),
        (1, {0: 10}, "All ratio values"),
        (1, {0: -10}, "All ratio values"),
    ],
)
def test_multi_step_ratio_scheduler_errors(epoch, steps, error_msg):
    with pytest.raises(nncf.InternalError, match=error_msg):
        multi_step_ratio_scheduler(epoch, steps=steps)


@pytest.mark.parametrize(
    "epoch,expected",
    [
        (0, 0.2),
        (2, 0.39),
        (4, 0.54),
        (5, 0.60),
        (7, 0.69),
        (8, 0.73),
        (10, 0.8),
        (20, 0.8),
    ],
)
def test_exponential_ratio_scheduler_values(epoch: int, expected: float):
    r = exponential_ratio_scheduler(epoch, initial_ratio=0.2, target_ratio=0.8, target_epoch=10)
    assert pytest.approx(r, 0.01) == expected


@pytest.mark.parametrize(
    "epoch,initial_ratio,target_ratio,target_epoch,error_msg",
    [
        (-1, 0.2, 0.8, 10, "Epoch"),
        (1, -0.1, 0.8, 10, "Initial ratio"),
        (1, 1.1, 0.8, 10, "Initial ratio"),
        (1, 0.8, 0.2, 10, "less than target"),
        (1, 0.2, 1.5, 10, "Target ratio"),
        (1, 0.2, 0.8, 0, "Total epochs"),
    ],
)
def test_exponential_ratio_scheduler_errors(epoch, initial_ratio, target_ratio, target_epoch, error_msg):
    with pytest.raises(nncf.InternalError, match=error_msg):
        exponential_ratio_scheduler(
            epoch, initial_ratio=initial_ratio, target_ratio=target_ratio, target_epoch=target_epoch
        )
