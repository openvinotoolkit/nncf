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
import pytest
import torch

from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from tests.cross_fw.test_templates.template_test_nncf_tensor import TemplateTestNNCFTensorOperators


def cast_to(x: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
    if dtype is TensorDataType.float32:
        return x.type(torch.float32)
    if dtype is TensorDataType.float16:
        return x.type(torch.float16)
    raise NotImplementedError


class TestPTNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        return torch.tensor(x)

    @staticmethod
    def to_cpu(x):
        return x

    @staticmethod
    def cast_to(x: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
        return cast_to(x, dtype)

    @staticmethod
    def backend() -> TensorBackend:
        return TensorBackend.torch

    @staticmethod
    def device() -> TensorDeviceType:
        return TensorDeviceType.CPU


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skipping for CPU-only setups")
class TestCudaPTNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        return torch.tensor(x).cuda()

    @staticmethod
    def to_cpu(x):
        return x.cpu()

    @staticmethod
    def cast_to(x: torch.Tensor, dtype: TensorDataType) -> torch.Tensor:
        return cast_to(x, dtype)

    def test_device(self):
        tensor = Tensor(self.to_tensor([1]))
        assert tensor.device == TensorDeviceType.GPU

    @staticmethod
    def backend() -> TensorBackend:
        return TensorBackend.torch

    @staticmethod
    def device() -> TensorDeviceType:
        return TensorDeviceType.GPU
