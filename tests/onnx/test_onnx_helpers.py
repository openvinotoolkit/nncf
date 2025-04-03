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
import pytest

from nncf.onnx.graph.onnx_helper import pack_int4
from nncf.onnx.graph.onnx_helper import pack_uint4
from nncf.onnx.graph.onnx_helper import unpack_int4
from nncf.onnx.graph.onnx_helper import unpack_uint4


@pytest.mark.parametrize("tensor", [np.random.randint(-8, 8, (8, 16, 32)).astype(np.int8)])
def test_int4_packing(tensor: np.ndarray):
    unpacked = unpack_int4(pack_int4(tensor))
    unpacked = unpacked.reshape(tensor.shape)
    assert np.allclose(tensor, unpacked)


@pytest.mark.parametrize("tensor", [np.random.randint(0, 16, (8, 16, 32)).astype(np.uint8)])
def test_uint4_packing(tensor: np.ndarray):
    unpacked = unpack_uint4(pack_uint4(tensor))
    unpacked = unpacked.reshape(tensor.shape)
    assert np.allclose(tensor, unpacked)
