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

import numpy as np

from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from tests.cross_fw.test_templates.template_test_nncf_tensor import TemplateTestNNCFTensorOperators


class TestNPNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        return np.array(x)

    @staticmethod
    def to_cpu(x):
        return x

    @staticmethod
    def cast_to(x: np.ndarray, dtype: TensorDataType) -> np.ndarray:
        if dtype is TensorDataType.float32:
            return x.astype(np.float32)
        if dtype is TensorDataType.float16:
            return x.astype(np.float16)
        raise NotImplementedError

    @staticmethod
    def backend() -> TensorBackend:
        return TensorBackend.numpy

    @staticmethod
    def device() -> TensorDeviceType:
        return TensorDeviceType.CPU
