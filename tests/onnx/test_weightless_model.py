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

import onnx
import pytest
import torch
from torchvision import models

from tests.onnx.quantization.common import ModelToTest
from tests.onnx.weightless_model import save_model_without_tensors


@pytest.mark.parametrize(
    ("model_to_test", "model"),
    [
        (ModelToTest("resnet18", [1, 3, 224, 224]), models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)),
    ],
)
def test_save_weightless_model(tmp_path, model_to_test, model):
    onnx_model_path = tmp_path / (model_to_test.model_name + ".onnx")
    x = torch.randn([1, 3, 224, 224], requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path)
    onnx_model = onnx.load_model(onnx_model_path)

    weightless_model_path = tmp_path / Path("weightless_model.onnx")
    save_model_without_tensors(onnx_model, weightless_model_path)
    assert weightless_model_path.stat().st_size < Path(onnx_model_path).stat().st_size
