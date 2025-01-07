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

from pathlib import Path
from typing import Tuple

import pytest
import timm
import torch
from torch import nn

from tests.torch.models_hub_test.common import BaseTestModel
from tests.torch.models_hub_test.common import ExampleType
from tests.torch.models_hub_test.common import ModelInfo
from tests.torch.models_hub_test.common import get_model_params
from tests.torch.models_hub_test.common import idfn

MODEL_LIST_FILE = Path(__file__).parent / "timm_models.txt"


class TestTimmModel(BaseTestModel):
    def load_model(self, model_name: str) -> Tuple[nn.Module, ExampleType]:
        m = timm.create_model(model_name, pretrained=False)
        cfg = timm.get_pretrained_cfg(model_name)
        shape = [1] + list(cfg.input_size)
        example = (torch.randn(shape),)
        return m, example

    @pytest.mark.parametrize("model_info", get_model_params(MODEL_LIST_FILE), ids=idfn)
    def test_nncf_wrap(self, model_info: ModelInfo):
        self.nncf_wrap(model_info.model_name)
