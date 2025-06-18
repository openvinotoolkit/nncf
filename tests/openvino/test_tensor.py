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
import openvino as ov
import pytest
import torch

import nncf.tensor.functions as fns
from nncf.tensor import Tensor
from nncf.tensor.definitions import TensorBackend

DATA = [1, 2, 3]


@pytest.mark.parametrize(
    "tensor",
    [
        Tensor(np.array(DATA, np.float32)),
        Tensor(torch.tensor(DATA, dtype=torch.float32)),
        Tensor(ov.Tensor(np.array(DATA, np.float32), (len(DATA),), ov.Type.f32)),
    ],
    ids=["from_numpy", "from_torch", "from_openvino"],
)
def test_as_openvino_tensor(tensor: Tensor):
    if tensor.backend == TensorBackend.torch:
        pytest.skip("PT -> OV conversion is currently not supported.")
    ov_tensor = tensor.as_openvino_tensor()
    assert ov_tensor.backend == TensorBackend.ov
    assert ov_tensor.dtype == tensor.dtype
    assert ov_tensor.shape == tensor.shape
    assert fns.allclose(tensor.as_numpy_tensor(), ov_tensor.as_numpy_tensor())
