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
from typing import Any, Dict, List, Union

from nncf import NNCFConfig
from nncf.api.compression import CompressionAlgorithmController
from nncf.api.compression import TModel
from nncf.common.composite_compression import CompositeCompressionAlgorithmBuilder
from nncf.common.compression import BaseCompressionAlgorithmBuilder
from nncf.common.graph.transformations.layout import TransformationLayout
from nncf.common.utils.registry import Registry

STATE_ATTR = "state"
DIFF_STATE_ATTR = STATE_ATTR + "__"


class A(BaseCompressionAlgorithmBuilder):
    def __init__(self, config: NNCFConfig, should_init: bool = True, state_value: int = 1, name: str = "A"):
        setattr(self, Registry.REGISTERED_NAME_ATTR, name)
        super().__init__(config, should_init)
        self.state_value = state_value

    def _load_state_without_name(self, state_without_name: Dict[str, Any]):
        self.state_value = state_without_name.get(STATE_ATTR)

    def _get_state_without_name(self) -> Dict[str, Any]:
        return {STATE_ATTR: self.state_value}

    def apply_to(self, model: TModel) -> TModel:
        pass

    def _build_controller(self, model: TModel) -> CompressionAlgorithmController:
        pass

    def get_transformation_layout(self, model: TModel) -> TransformationLayout:
        pass

    def initialize(self, model: TModel) -> None:
        pass


class CA(CompositeCompressionAlgorithmBuilder):
    @property
    def name(self) -> str:
        pass

    def add(self, child_builder) -> None:
        self._child_builders.append(child_builder)

    def apply_to(self, model: TModel) -> TModel:
        pass

    def build_controller(self, model: TModel) -> CompressionAlgorithmController:
        pass

    def get_transformation_layout(self, model: TModel) -> TransformationLayout:
        pass

    def initialize(self, model: TModel) -> None:
        pass


def _get_mock_config(algo_name: Union[List[str], str]) -> NNCFConfig:
    config = NNCFConfig()
    config["input_info"] = {"sample_size": [1, 1]}
    if isinstance(algo_name, list):
        lst = []
        for alg_n in algo_name:
            lst.append({"algorithm": alg_n})
        config["compression"] = lst
    else:
        assert isinstance(algo_name, str)
        config["compression"] = {"algorithm": algo_name}
    return config


def test_builder_state_load():
    config = _get_mock_config("A")
    builder = A(config, True, 1)
    builder.state_value += 1

    saved_state = builder.get_state()

    builder = A(config, True, 1)
    builder.load_state(saved_state)

    assert builder.state_value == 2


def test_basic_composite_builder_load():
    def create_builder():
        config = _get_mock_config(["A", "A2"])
        c = CA(config, True)
        a = A(config, True, 1)
        b = A(config, True, 2, "A2")
        c.add(a)
        c.add(b)
        return c, a, b

    composite_bldr, bldr1, bldr2 = create_builder()

    bldr1.state_value += 1
    bldr2.state_value += 2

    saved_state = composite_bldr.get_state()

    composite_bldr, bldr1, bldr2 = create_builder()

    composite_bldr.load_state(saved_state)

    assert bldr1.state_value == 2
    assert bldr2.state_value == 4


def test_advanced_composite_ctrl_load():
    config = _get_mock_config(["A", "A2", "A3"])
    composite_builder = CA(config, True)
    ctrl1 = A(config, True, 1)
    ctrl2 = A(config, True, 2, name="A2")
    composite_builder.add(ctrl1)
    composite_builder.add(ctrl2)

    ctrl1.state_value += 1
    ctrl2.state_value += 2

    saved_state = composite_builder.get_state()

    composite_builder = CA(config, True)
    ctrl1 = A(config, True, 1)
    ctrl3 = A(config, True, 3, name="A3")
    composite_builder.add(ctrl1)
    composite_builder.add(ctrl3)

    composite_builder.load_state(saved_state)

    assert ctrl1.state_value == 2
    assert ctrl3.state_value == 3
