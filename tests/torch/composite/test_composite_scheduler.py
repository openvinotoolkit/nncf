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
from typing import Optional

from nncf.api.compression import CompressionScheduler
from nncf.common.composite_compression import CompositeCompressionScheduler


class DummyScheduler(CompressionScheduler):
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def load_state(self, state):
        self.delta = state["delta"]

    def get_state(self):
        state = {"delta": self.delta}
        return state

    def step(self, next_step: Optional[int] = None) -> None:
        pass

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        pass


def test_can_restore_from_state():
    def _create_composite_scheduler(deltas):
        schedulers = [DummyScheduler(delta) for delta in deltas]
        composite_scheduler = CompositeCompressionScheduler()
        for scheduler in schedulers:
            composite_scheduler.add(scheduler)
        return composite_scheduler

    expected = _create_composite_scheduler([1, 2])
    actual = _create_composite_scheduler([0, 0])
    actual.load_state(expected.get_state())

    for expected_child, actual_child in zip(expected.child_schedulers, actual.child_schedulers):
        assert expected_child.delta == actual_child.delta
