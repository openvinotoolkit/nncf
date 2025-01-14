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
from typing import Any, Dict, List, Optional, Tuple

from nncf.api.compression import CompressionLoss
from nncf.api.compression import CompressionScheduler
from nncf.api.compression import CompressionStage
from nncf.api.statistics import Statistics
from nncf.common.composite_compression import CompositeCompressionAlgorithmController
from nncf.common.compression import BaseCompressionAlgorithmController
from nncf.common.statistics import NNCFStatistics

STATE_ATTR = "state"
DIFF_STATE_ATTR = STATE_ATTR + "__"


class ALoss(CompressionLoss):
    def __init__(self, state_value):
        self.state = state_value

    def calculate(self, *args, **kwargs) -> Any:
        pass

    def load_state(self, state: Dict[str, Any]) -> None:
        self.state = state.get(STATE_ATTR)

    def get_state(self) -> Dict[str, Any]:
        return {STATE_ATTR: self.state}


class BLoss(ALoss):
    def load_state(self, state: Dict[str, Any]) -> None:
        self.state = state.get(DIFF_STATE_ATTR)

    def get_state(self) -> Dict[str, Any]:
        return {DIFF_STATE_ATTR: self.state}


class AScheduler(CompressionScheduler):
    def __init__(self, state_value):
        self.state = state_value

    def step(self, next_step: Optional[int] = None) -> None:
        pass

    def epoch_step(self, next_epoch: Optional[int] = None) -> None:
        pass

    def load_state(self, state: Dict[str, Any]) -> None:
        self.state = state.get(STATE_ATTR)

    def get_state(self) -> Dict[str, Any]:
        return {STATE_ATTR: self.state}


class BScheduler(AScheduler):
    def load_state(self, state: Dict[str, Any]) -> None:
        self.state = state.get(DIFF_STATE_ATTR)

    def get_state(self) -> Dict[str, Any]:
        return {DIFF_STATE_ATTR: self.state}


class A(BaseCompressionAlgorithmController):
    def __init__(self, target_model, state_value: int, name: str = "A"):
        super().__init__(target_model)
        self._state_value = state_value
        self._loss = ALoss(self._state_value)
        self._scheduler = AScheduler(self._state_value)
        self._name = name

    @property
    def loss(self) -> CompressionLoss:
        return self._loss

    @property
    def scheduler(self) -> CompressionScheduler:
        return self._scheduler

    def get_compression_state(self) -> Dict[str, Any]:
        pass

    def statistics(self, quickly_collected_only: bool = False) -> Statistics:
        return NNCFStatistics()

    def compression_stage(self) -> CompressionStage:
        pass


class B(A):
    def __init__(self, target_model, state_value: int):
        super().__init__(target_model, state_value, name="B")
        self._loss = BLoss(self._state_value)
        self._scheduler = BScheduler(self._state_value)


class CA(CompositeCompressionAlgorithmController):
    @property
    def compression_rate(self) -> float:
        pass

    def disable_scheduler(self) -> None:
        pass

    @property
    def name(self) -> str:
        pass

    def get_compression_state(self) -> Dict[str, Any]:
        pass

    def export_model(
        self,
        save_path: str,
        save_format: Optional[str] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        model_args: Optional[Tuple[Any, ...]] = None,
    ) -> None:
        pass


def test_ctrl_state_load(mocker):
    model = mocker.stub
    ctrl = A(model, 1)
    ctrl.loss.state += 1
    ctrl.scheduler.state += 1

    saved_state = ctrl.get_state()

    ctrl = A(model, 1)
    ctrl.load_state(saved_state)

    assert ctrl.loss.state == 2
    assert ctrl.scheduler.state == 2


def test_basic_composite_ctrl_load(mocker):
    def create_ctrl():
        model = mocker.stub
        c = CA(model)
        a = A(model, 1)
        b = B(model, 2)
        c.add(a)
        c.add(b)
        return c, a, b

    composite_ctrl, ctrl1, ctrl2 = create_ctrl()

    ctrl1.loss.state += 1
    ctrl1.scheduler.state += 1
    ctrl2.loss.state += 2
    ctrl2.scheduler.state += 2

    saved_state = composite_ctrl.get_state()

    composite_ctrl, ctrl1, ctrl2 = create_ctrl()

    composite_ctrl.load_state(saved_state)

    assert ctrl1.loss.state == 2
    assert ctrl1.scheduler.state == 2
    assert ctrl2.loss.state == 4
    assert ctrl2.scheduler.state == 4


def test_advanced_composite_ctrl_load(mocker):
    model = mocker.stub
    composite_ctrl = CA(model)
    ctrl1 = A(model, 1)
    ctrl2 = A(model, 2, name="A2")
    composite_ctrl.add(ctrl1)
    composite_ctrl.add(ctrl2)

    ctrl1.loss.state += 1
    ctrl1.scheduler.state += 1
    ctrl2.loss.state += 2
    ctrl2.scheduler.state += 2

    saved_state = composite_ctrl.get_state()

    composite_ctrl = CA(model)
    ctrl1 = A(model, 1)
    ctrl3 = B(model, 3)
    composite_ctrl.add(ctrl1)
    composite_ctrl.add(ctrl3)

    composite_ctrl.load_state(saved_state)

    assert ctrl1.loss.state == 2
    assert ctrl1.scheduler.state == 2
    assert ctrl3.loss.state == 3
    assert ctrl3.scheduler.state == 3
