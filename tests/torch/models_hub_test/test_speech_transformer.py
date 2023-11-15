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
import tempfile

import torch

from tests.torch.models_hub_test.common import BaseTestModel


class TestSpeechTransformerModel(BaseTestModel):
    def load_model(self, model_name: str):
        self.repo_dir = tempfile.TemporaryDirectory()
        os.system(f"git clone https://github.com/mvafin/Speech-Transformer.git {self.repo_dir.name}")
        subprocess.check_call(["git", "checkout", "071eebb7549b66bae2cb93e3391fe99749389456"], cwd=self.repo_dir.name)

        sys.path.append(self.repo_dir.name)
        from transformer.transformer import Transformer

        m = Transformer()

        example = (
            torch.randn(32, 209, 320),
            torch.stack(sorted(torch.randint(55, 250, [32]), reverse=True)),
            torch.randint(-1, 4232, [32, 20]),
        )
        return m, example

    def test_nncf_wrap(self):
        self.nncf_wrap("speech-transformer")
