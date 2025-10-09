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
import openvino.opset13 as opset
import pytest

import nncf.tensor.functions as fns
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.functions.numpy_numeric import DTYPE_MAP as DTYPE_MAP_NP
from nncf.tensor.functions.openvino_numeric import DTYPE_MAP as DTYPE_MAP_OV


class TestOVNNCFTensorOperators:
    @staticmethod
    def to_tensor(x, backend=TensorBackend.ov, dtype=TensorDataType.float32):
        no_numpy_support_dtypes = [
            TensorDataType.bfloat16,
            TensorDataType.uint4,
            TensorDataType.int4,
            TensorDataType.nf4,
            TensorDataType.f8e5m2,
            TensorDataType.f8e4m3,
        ]

        if backend == TensorBackend.ov:
            if dtype in no_numpy_support_dtypes:
                ov_const = opset.constant(x, dtype=DTYPE_MAP_OV[dtype])
                return ov.Tensor(ov_const.data, ov_const.data.shape, DTYPE_MAP_OV[dtype])
            return ov.Tensor(np.array(x, dtype=DTYPE_MAP_NP[dtype]))
        if backend == TensorBackend.numpy:
            if dtype in no_numpy_support_dtypes:
                msg = f"Can't create NumPY tensor in dtype {dtype}"
                raise ValueError(msg)
            return np.array(x, dtype=DTYPE_MAP_NP[dtype])
        msg = "Unsupported backend"
        raise ValueError(msg)

    @staticmethod
    def backend() -> TensorBackend:
        return TensorBackend.ov

    def test_property_backend(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        assert tensor_a.backend == self.backend()

    def test_device(self):
        tensor = Tensor(self.to_tensor([1]))
        assert tensor.device == TensorDeviceType.CPU

    def test_size(self):
        tensor = Tensor(self.to_tensor([1, 1]))
        res = tensor.size
        assert res == 2

    def test_astype(self):
        tensor = Tensor(self.to_tensor([1]))
        res = tensor.astype(TensorDataType.int8)
        assert isinstance(res, Tensor)
        assert res.dtype == TensorDataType.int8
        assert res.device == tensor.device

    def test_fn_astype(self):
        tensor = Tensor(self.to_tensor([1]))
        res = fns.astype(tensor, TensorDataType.int8)
        assert isinstance(res, Tensor)
        assert res.dtype == TensorDataType.int8

    def test_reshape(self):
        tensor = Tensor(self.to_tensor([1, 1]))
        res = tensor.reshape((1, 2))
        assert tensor.shape == (2,)
        assert res.shape == (1, 2)
        assert res.device == tensor.device

    def test_fn_reshape(self):
        tensor = Tensor(self.to_tensor([1, 1]))
        res = fns.reshape(tensor, (1, 2))
        assert tensor.shape == (2,)
        assert res.shape == (1, 2)
        assert res.device == tensor.device

    @pytest.mark.parametrize("from_backend", [TensorBackend.numpy, TensorBackend.ov])
    def test_as_numpy_tensor(self, from_backend):
        tensor1 = Tensor(self.to_tensor([1], backend=from_backend))
        assert tensor1.backend == from_backend
        tensor2 = tensor1.as_numpy_tensor()
        assert tensor2.backend == TensorBackend.numpy
        assert tensor1.dtype == tensor2.dtype
        assert tensor1.shape == tensor2.shape
        assert tensor1.device == tensor2.device
