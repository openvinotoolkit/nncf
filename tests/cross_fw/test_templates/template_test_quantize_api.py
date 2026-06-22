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
from abc import ABC
from abc import abstractmethod
from typing import TypeVar

import pytest

import nncf
from nncf.data.dataset import Dataset
from nncf.quantization.advanced_parameters import AdvancedQuantizationParameters

TModel = TypeVar("TModel")


class TemplateTestQuantizeApi(ABC):
    @staticmethod
    @abstractmethod
    def get_simple_model() -> TModel:
        """Returns a minimal model for the backend."""

    def test_quantize_calibration_device(self):
        model = self.get_simple_model()
        with pytest.raises(nncf.ParameterNotSupportedError):
            nncf.quantize(
                model,
                Dataset([0]),
                advanced_parameters=AdvancedQuantizationParameters(calibration_device="SOME_DEVICE"),
            )
