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
import pytest

import nncf
from nncf import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters
from tests.cross_fw.test_templates.template_test_quantize_api import TemplateTestQuantizeApi
from tests.onnx.models import LinearModel

INPUT_SHAPE = [1, 3, 32, 32]


class TestONNXQuantizeApi(TemplateTestQuantizeApi):
    @staticmethod
    def get_simple_model():
        return LinearModel().onnx_model

    def test_quantize_with_accuracy_control_calibration_device(self):
        model = self.get_simple_model()
        dataset = Dataset([np.ones(INPUT_SHAPE, dtype=np.float32)])
        with pytest.raises(nncf.ParameterNotSupportedError):
            nncf.quantize_with_accuracy_control(
                model,
                dataset,
                dataset,
                lambda model, dataset: (1.0, None),
                advanced_quantization_parameters=AdvancedQuantizationParameters(calibration_device="SOME_DEVICE"),
            )
