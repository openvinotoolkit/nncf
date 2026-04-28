# Copyright (c) 2026 Intel Corporation
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
from openvino import Model
from openvino import Shape
from openvino import Type
from openvino import op
from openvino import opset13 as opset

import nncf
from nncf import Dataset
from tests.cross_fw.shared.datasets import MockDataset

INPUT_SHAPE = [2, 1, 1, 1]


def get_mock_model() -> Model:
    param_node = op.Parameter(Type.f32, Shape(INPUT_SHAPE))
    softmax_axis = 1
    softmax_node = opset.softmax(param_node, softmax_axis)
    return Model(softmax_node, [param_node], "mock")


def test_non_positive_subset_size():
    model_to_test = get_mock_model()

    with pytest.raises(nncf.ValidationError) as e:
        nncf.quantize(model_to_test, Dataset(MockDataset(INPUT_SHAPE)), subset_size=0)
        assert "Subset size must be positive." in e.info


def test_quantize_calibration_device(monkeypatch):
    import numpy as np
    import openvino as ov

    from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
    from tests.openvino.native.models import LinearModel

    model_to_test = LinearModel().ov_model
    input_shape = [inp.shape for inp in model_to_test.inputs][0]
    dataset = Dataset([np.random.rand(*input_shape).astype(np.float32) for _ in range(2)])
    captured_devices = []

    original_compile = ov.Core.compile_model

    def mock_compile(self, model, device_name="CPU", config=None):
        captured_devices.append(device_name)
        return original_compile(self, model, device_name="CPU", config=config)

    monkeypatch.setattr(ov.Core, "compile_model", mock_compile)
    nncf.quantize(
        model_to_test,
        dataset,
        advanced_parameters=AdvancedQuantizationParameters(calibration_device="GPU"),
    )
    assert any(d == "GPU" for d in captured_devices)
