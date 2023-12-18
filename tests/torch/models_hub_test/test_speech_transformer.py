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

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

from tests.torch.models_hub_test.common import BaseTestModel
from tests.torch.models_hub_test.common import ExampleType


class TestSpeechTransformerModel(BaseTestModel):
    def load_model(self, model_name: str) -> Tuple[nn.Module, ExampleType]:
        os.system(f"git clone https://github.com/mvafin/Speech-Transformer.git {self.tmp_path}")
        subprocess.check_call(["git", "checkout", "071eebb7549b66bae2cb93e3391fe99749389456"], cwd=self.tmp_path)

        sys.path.append(self.tmp_path.as_posix())
        from transformer.transformer import Transformer

        m = Transformer()

        sys.path.remove(self.tmp_path.as_posix())
        example = (
            torch.randn(32, 209, 320),
            torch.stack(sorted(torch.randint(55, 250, [32]), reverse=True)),
            torch.randint(-1, 4232, [32, 20]),
        )
        return m, example

    def test_nncf_wrap(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.nncf_wrap("speech-transformer")
