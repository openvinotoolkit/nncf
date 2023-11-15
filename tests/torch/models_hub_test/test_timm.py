# Copyright (c) 2023 Intel Corporation
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
import timm
import torch

from tests.torch.models_hub_test.common import BaseTestModel


def filter_timm(timm_list: list) -> list:
    unique_models = set()
    filtered_list = []
    ignore_set = {
        "base", "mini", "small", "xxtiny", "xtiny", "tiny", "lite", "nano", "pico", "medium", "big",
        "large", "xlarge", "xxlarge", "huge", "gigantic", "giant", "enormous", "xs", "xxs", "s", "m", "l", "xl"
    }  # fmt: skip
    for name in timm_list:
        # first: remove datasets
        name_parts = name.split(".")
        _name = "_".join(name.split(".")[:-1]) if len(name_parts) > 1 else name
        # second: remove sizes
        name_set = set([n for n in _name.split("_") if not n.isnumeric()])
        name_set = name_set.difference(ignore_set)
        name_join = "_".join(name_set)
        if name_join not in unique_models:
            unique_models.add(name_join)
            filtered_list.append(name)
    return filtered_list


def get_all_models() -> list:
    m_list = timm.list_pretrained()
    return filter_timm(m_list)


@pytest.mark.models_hub
class TestTimmModel(BaseTestModel):
    def load_model(self, model_name: str):
        m = timm.create_model(model_name, pretrained=False)
        cfg = timm.get_pretrained_cfg(model_name)
        shape = [1] + list(cfg.input_size)
        example = (torch.randn(shape),)
        return m, example

    @pytest.mark.parametrize("name", get_all_models())
    def test_nncf_wrap(self, name):
        self.nncf_wrap(name)
