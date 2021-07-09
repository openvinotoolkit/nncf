"""
 Copyright (c) 2021 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
from typing import Any
from typing import Dict

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import ModelType
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.registry import Registry

STATE_ATTR = 'state'
DIFF_STATE_ATTR = STATE_ATTR + '__'


class A(BaseCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True, state_value: int = 1, name: str = 'A'):
        super().__init__(config, should_init)
        self.state_value = state_value
        setattr(self, Registry.REGISTERED_NAME_ATTR, name)

    def _load_state_without_name(self, state: Dict[str, Any]):
        self.state_value = state.get(STATE_ATTR)

    def _get_state_without_name(self) -> Dict[str, Any]:
        return {STATE_ATTR: self.state_value}

    def apply_to(self, model: ModelType) -> ModelType:
        pass

    def _build_controller(self, model: ModelType) -> CompressionAlgorithmController:
        pass

    def get_transformation_layout(self, model: ModelType) -> TransformationLayout:
        pass

    def initialize(self, model: ModelType) -> None:
        pass


class CA(CompositeCompressionAlgorithmBuilder):
    @property
    def name(self) -> str:
        pass

    def add(self, child_builder) -> None:
        self._child_builders.append(child_builder)

    def apply_to(self, model: ModelType) -> ModelType:
        pass

    def build_controller(self, model: ModelType) -> CompressionAlgorithmController:
        pass

    def get_transformation_layout(self, model: ModelType) -> TransformationLayout:
        pass

    def initialize(self, model: ModelType) -> None:
        pass


def test_builder_state_load(mocker):
    config = mocker.stub
    builder = A(config, True, 1)
    builder.state_value += 1

    saved_state = builder.get_state()

    builder = A(config, True, 1)
    builder.load_state(saved_state)

    assert builder.state_value == 2


def test_basic_composite_builder_load(mocker):
    def create_builder():
        config = mocker.stub
        c = CA(config, True)
        a = A(config, True, 1)
        b = A(config, True, 2, 'A2')
        c.add(a)
        c.add(b)
        return c, a, b

    composite_ctrl, ctrl1, ctrl2 = create_builder()

    ctrl1.state_value += 1
    ctrl2.state_value += 2

    saved_state = composite_ctrl.get_state()

    composite_ctrl, ctrl1, ctrl2 = create_builder()

    composite_ctrl.load_state(saved_state)

    assert ctrl1.state_value == 2
    assert ctrl2.state_value == 4


def test_advanced_composite_ctrl_load(mocker):
    config = mocker.stub
    composite_builder = CA(config, True)
    ctrl1 = A(config, True, 1)
    ctrl2 = A(config, True, 2, name='A2')
    composite_builder.add(ctrl1)
    composite_builder.add(ctrl2)

    ctrl1.state_value += 1
    ctrl2.state_value += 2

    saved_state = composite_builder.get_state()

    composite_builder = CA(config, True)
    ctrl1 = A(config, True, 1)
    ctrl3 = A(config, True, 3, name='A3')
    composite_builder.add(ctrl1)
    composite_builder.add(ctrl3)

    composite_builder.load_state(saved_state)

    assert ctrl1.state_value == 2
    assert ctrl3.state_value == 3
