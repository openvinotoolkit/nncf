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
import numpy as np
import openvino as ov
import pytest

import nncf
from nncf import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from tests.cross_fw.test_templates.template_test_quantize_api import TemplateTestQuantizeApi
from tests.openvino.native.models import LinearModel

LINEAR_MODEL_INPUT_SHAPE = [1, 3, 4, 2]


class TestOVQuantizeApi(TemplateTestQuantizeApi):
    @staticmethod
    def get_simple_model() -> ov.Model:
        return LinearModel().ov_model

    def test_quantize_calibration_device(self, monkeypatch):
        model = self.get_simple_model()
        dataset = Dataset([np.ones(LINEAR_MODEL_INPUT_SHAPE, dtype=np.float32)])
        captured_devices = []

        original_compile = ov.Core.compile_model

        def mock_compile(self, model, device_name="CPU", config=None):
            captured_devices.append(device_name)
            return original_compile(self, model, device_name="CPU", config=config)

        monkeypatch.setattr(ov.Core, "compile_model", mock_compile)
        nncf.quantize(
            model,
            dataset,
            advanced_parameters=AdvancedQuantizationParameters(calibration_device="SOME_DEVICE"),
        )
        assert all(d == "SOME_DEVICE" for d in captured_devices)

    def test_quantize_with_accuracy_control_calibration_device(self, monkeypatch):
        model = self.get_simple_model()
        dataset = Dataset([np.ones(LINEAR_MODEL_INPUT_SHAPE, dtype=np.float32)])
        captured_devices = []

        original_compile = ov.Core.compile_model

        def mock_compile(self, model, device_name="CPU", config=None):
            captured_devices.append(device_name)
            return original_compile(self, model, device_name="CPU", config=config)

        monkeypatch.setattr(ov.Core, "compile_model", mock_compile)
        nncf.quantize_with_accuracy_control(
            model,
            dataset,
            dataset,
            lambda model, dataset: (1.0, None),
            advanced_quantization_parameters=AdvancedQuantizationParameters(calibration_device="SOME_DEVICE"),
        )
        assert "SOME_DEVICE" in captured_devices

    def test_non_positive_subset_size(self):
        model_to_test = self.get_simple_model()

        with pytest.raises(nncf.ValidationError) as e:
            nncf.quantize(model_to_test, Dataset([np.ones(LINEAR_MODEL_INPUT_SHAPE, dtype=np.float32)]), subset_size=0)
            assert "Subset size must be positive." in e.info
