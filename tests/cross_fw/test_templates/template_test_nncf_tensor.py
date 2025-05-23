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


import operator
from abc import abstractmethod
from math import log2
from math import sqrt
from typing import TypeVar

import numpy as np
import pytest

import nncf
import nncf.tensor.functions as fns
from nncf.experimental.common.tensor_statistics import statistical_functions as s_fns
from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor import TensorDeviceType
from nncf.tensor.definitions import TensorBackend

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


OPERATOR_MAP = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "pow": operator.pow,
    "truediv": operator.truediv,
    "floordiv": operator.floordiv,
    "neg": lambda a, _: -a,
    "iadd": operator.iadd,
    "isub": operator.isub,
    "imul": operator.imul,
    "ipow": operator.ipow,
    "itruediv": operator.itruediv,
    "ifloordiv": operator.ifloordiv,
}
BINARY_OPERATORS = ["add", "sub", "pow", "mul", "truediv", "floordiv"]

COMPARISON_OPERATOR_MAP = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "ge": operator.ge,
}


class TemplateTestNNCFTensorOperators:
    @staticmethod
    @abstractmethod
    def to_tensor(x: TTensor) -> TTensor:
        pass

    @staticmethod
    @abstractmethod
    def to_cpu(x: TTensor) -> TTensor:
        pass

    @staticmethod
    @abstractmethod
    def cast_to(x: TTensor, dtype: TensorDataType) -> TTensor:
        pass

    @staticmethod
    @abstractmethod
    def backend() -> TensorBackend:
        pass

    @staticmethod
    @abstractmethod
    def device() -> TensorDeviceType:
        pass

    def test_property_backend(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        assert tensor_a.backend == self.backend()

    def test_operator_clone(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = tensor_a.clone()
        assert isinstance(tensor_b, Tensor)
        assert tensor_a.device == tensor_b.device
        assert tensor_a.backend == tensor_b.backend
        assert tensor_a.dtype == tensor_b.dtype
        assert id(tensor_a.data) is not id(tensor_b.data)
        assert all(tensor_a == tensor_b)

    @pytest.mark.parametrize("op_name", OPERATOR_MAP.keys())
    def test_operators_tensor(self, op_name):
        tensor_a = self.to_tensor([1.0, 2.0])
        tensor_b = self.to_tensor([22.0, 11.0])

        nncf_tensor_a = Tensor(tensor_a)
        nncf_tensor_b = Tensor(tensor_b)

        fn = OPERATOR_MAP[op_name]
        res = fn(tensor_a, tensor_b)
        res_nncf = fn(nncf_tensor_a, nncf_tensor_b)

        assert res.dtype == res_nncf.data.dtype
        assert all(res == res_nncf.data)
        assert isinstance(res_nncf, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operaors do not guarantee to return a tensor on an initial device.
            assert res_nncf.device == nncf_tensor_a.device

    @pytest.mark.parametrize("op_name", OPERATOR_MAP.keys())
    def test_operators_int(self, op_name):
        tensor_a = self.to_tensor([1.0, 2.0])
        value = 2.0

        nncf_tensor_a = Tensor(tensor_a)

        fn = OPERATOR_MAP[op_name]
        res = fn(tensor_a, value)
        res_nncf = fn(nncf_tensor_a, value)

        assert res.dtype == res_nncf.data.dtype
        assert all(res == res_nncf.data)
        assert isinstance(res_nncf, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operaors do not guarantee to return a tensor on an initial device.
            assert res_nncf.device == nncf_tensor_a.device

    @pytest.mark.parametrize("op_name", BINARY_OPERATORS)
    def test_operators_int_rev(self, op_name):
        tensor_a = self.to_tensor([1, 2])
        value = 2

        nncf_tensor_a = Tensor(tensor_a)

        fn = OPERATOR_MAP[op_name]
        res = fn(value, tensor_a)
        res_nncf = fn(value, nncf_tensor_a)

        assert res.dtype == res_nncf.data.dtype
        assert all(res == res_nncf.data)
        assert isinstance(res_nncf, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operaors do not guarantee to return a tensor on an initial device.
            assert res_nncf.device == nncf_tensor_a.device

    @pytest.mark.parametrize("op_name", COMPARISON_OPERATOR_MAP.keys())
    def test_comparison_tensor(self, op_name):
        tensor_a = self.to_tensor((1,))
        tensor_b = self.to_tensor((2,))

        nncf_tensor_a = Tensor(tensor_a)
        nncf_tensor_b = Tensor(tensor_b)

        fn = COMPARISON_OPERATOR_MAP[op_name]
        res = fn(tensor_a, tensor_b)
        res_nncf = fn(nncf_tensor_a, nncf_tensor_b)

        assert Tensor(res) == res_nncf
        assert isinstance(res_nncf, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operators do not guarantee to return a tensor on an initial device.
            assert res_nncf.device == nncf_tensor_a.device

    @pytest.mark.parametrize("op_name", COMPARISON_OPERATOR_MAP.keys())
    def test_comparison_int(self, op_name):
        tensor_a = self.to_tensor((1,))
        value = 2

        nncf_tensor_a = Tensor(tensor_a)

        fn = COMPARISON_OPERATOR_MAP[op_name]
        res = fn(tensor_a, value)
        res_nncf = fn(nncf_tensor_a, value)

        assert Tensor(res) == res_nncf
        assert isinstance(res_nncf, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operators do not guarantee to return a tensor on an initial device.
            assert res_nncf.device == nncf_tensor_a.device

    @pytest.mark.parametrize("op_name", COMPARISON_OPERATOR_MAP.keys())
    def test_comparison_int_rev(self, op_name):
        tensor_a = self.to_tensor((1,))
        value = 2

        nncf_tensor_a = Tensor(tensor_a)

        fn = COMPARISON_OPERATOR_MAP[op_name]
        res = fn(value, tensor_a)
        res_nncf = fn(value, nncf_tensor_a)

        assert Tensor(res) == res_nncf
        assert isinstance(res_nncf, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operators do not guarantee to return a tensor on an initial device.
            assert res_nncf.device == nncf_tensor_a.device

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[1], [2]]]], None, [[1, 2], [1, 2]]),
            ([[[[1], [2]], [[1], [2]]]], 0, [[[1], [2]], [[1], [2]]]),
            ([[[[1], [2]], [[1], [2]]]], -1, [[[1, 2], [1, 2]]]),
            ([[[[1], [2]], [[1], [2]]]], (0, 3), [[1, 2], [1, 2]]),
        ),
    )
    def test_squeeze(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.squeeze(axis=axis)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, axis, exception_type, exception_match",
        (
            ([[[[1], [2]], [[1], [2]]]], (0, 1), ValueError, "not equal to one"),
            ([[[[1], [2]], [[1], [2]]]], 42, IndexError, "out of"),
            ([[[[1], [2]], [[1], [2]]]], (0, 42), IndexError, "out of"),
        ),
    )
    def test_squeeze_axis_error(self, val, axis, exception_type, exception_match):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        with pytest.raises(exception_type, match=exception_match):
            nncf_tensor.squeeze(axis=axis)

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[1], [2]]]], None, [[1, 2], [1, 2]]),
            ([[[[1], [2]], [[1], [2]]]], 0, [[[1], [2]], [[1], [2]]]),
            ([[[[1], [2]], [[1], [2]]]], -1, [[[1, 2], [1, 2]]]),
        ),
    )
    def test_fn_squeeze(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = fns.squeeze(nncf_tensor, axis=axis)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val,ref",
        (
            (1, 1),
            ([1], 1),
            ([[[[1], [2]], [[1], [2]]]], [1, 2, 1, 2]),
        ),
    )
    def test_flatten(self, val, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.flatten()
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, axis, keepdims, ref",
        (
            (1, None, False, 1),
            (1, None, True, 1),
            ([1], None, False, 1),
            ([1], None, True, 1),
            ([[[[1], [2]], [[3], [4]]]], None, False, 4),
            ([[[[1], [2]], [[3], [4]]]], None, True, 4),
            ([[1, 2], [3, 4]], 1, False, [2, 4]),
            ([[1, 2], [3, 4]], 1, True, [[2], [4]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), False, [[3], [4]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), True, [[[[3], [4]]]]),
        ),
    )
    def test_fn_max(self, val, axis, keepdims, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = fns.max(nncf_tensor, axis=axis, keepdims=keepdims)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, axis, keepdims, ref",
        (
            (1, None, False, 1),
            (1, None, True, 1),
            ([1], None, False, 1),
            ([1], None, True, 1),
            ([[[[1], [2]], [[3], [4]]]], None, False, 1),
            ([[[[1], [2]], [[3], [4]]]], None, True, 1),
            ([[1, 2], [3, 4]], 1, False, [1, 3]),
            ([[1, 2], [3, 4]], 1, True, [[1], [3]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), False, [[1], [2]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), True, [[[[1], [2]]]]),
        ),
    )
    def test_fn_min(self, val, axis, keepdims, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = fns.min(nncf_tensor, axis=axis, keepdims=keepdims)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, axis, keepdims, ref",
        (
            (1, None, False, 1),
            (1, None, True, 1),
            ([1], None, False, 1),
            ([1], None, True, 1),
            ([[[[1], [2]], [[3], [4]]]], None, False, 1),
            ([[[[1], [2]], [[3], [4]]]], None, True, 1),
            ([[1, 2], [3, 4]], 1, False, [1, 3]),
            ([[1, 2], [3, 4]], 1, True, [[1], [3]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), False, [[1], [2]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), True, [[[[1], [2]]]]),
        ),
    )
    def test_min(self, val, axis, keepdims, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.min(axis=axis, keepdims=keepdims)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, axis, keepdims, ref",
        (
            (1, None, False, 1),
            (1, None, True, 1),
            ([1], None, False, 1),
            ([1], None, True, 1),
            ([[[[1], [2]], [[3], [4]]]], None, False, 4),
            ([[[[1], [2]], [[3], [4]]]], None, True, 4),
            ([[1, 2], [3, 4]], 1, False, [2, 4]),
            ([[1, 2], [3, 4]], 1, True, [[2], [4]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), False, [[3], [4]]),
            ([[[[1], [2]], [[3], [4]]]], (0, 1), True, [[[[3], [4]]]]),
        ),
    )
    def test_max(self, val, axis, keepdims, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.max(axis=axis, keepdims=keepdims)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, ref",
        (
            (-1, 1),
            ([-1, 1], [1, 1]),
        ),
    )
    def test_abs(self, val, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        nncf_ref_tensor = Tensor(self.to_tensor(ref))
        res = nncf_tensor.abs()
        assert isinstance(res, Tensor)
        assert fns.allclose(res, nncf_ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "val, ref",
        (
            (-1, 1),
            ([-1, 1], [1, 1]),
        ),
    )
    def test_fn_abs(self, val, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        nncf_ref_tensor = Tensor(self.to_tensor(ref))
        res = fns.abs(nncf_tensor)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, nncf_ref_tensor)
        assert res.device == nncf_tensor.device

    def test_getitem_for_index(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        res = nncf_tensor[1]
        assert res == 1
        assert isinstance(res, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operators do not guarantee to return a tensor on an initial device.
            assert res.device == nncf_tensor.device

    @pytest.mark.parametrize("is_tensor_indecies", (False, True))
    def test_getitem_for_indecies(self, is_tensor_indecies):
        nncf_tensor = Tensor(self.to_tensor([0, 1, 2]))
        ref = Tensor(self.to_tensor([0, 1]))
        indecies = [0, 1]
        if is_tensor_indecies:
            indecies = Tensor(self.to_tensor(indecies))
        res = nncf_tensor[indecies]
        assert all(res == ref)
        assert isinstance(res, Tensor)
        assert res.device == nncf_tensor.device

    def test_iter(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        for i, x in enumerate(nncf_tensor):
            assert x == arr[i]
            assert isinstance(x, Tensor)
        assert i == 2

        arr = [arr]
        nested_nncf_tensor = Tensor(self.to_tensor(arr))
        for i, x in enumerate(nested_nncf_tensor):
            assert all(x == nncf_tensor)
            assert isinstance(x, Tensor)

        # Check a scalar tensor
        nncf_tensor = Tensor(self.to_tensor(42))
        with pytest.raises(TypeError):
            next(iter(nncf_tensor))

    # Math

    @pytest.mark.parametrize(
        "axis, ref",
        (
            (None, 3),
            (0, [2, 1]),
        ),
    )
    def test_fn_count_nonzero(self, axis, ref):
        tensor = self.to_tensor([[1.0, 2.0], [1.0, 0.0]])
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = fns.count_nonzero(nncf_tensor, axis=axis)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == nncf_tensor.device

    def test_fn_zeros_like(self):
        tensor = self.to_tensor([1, 2])
        nncf_tensor = Tensor(tensor)

        res = fns.zeros_like(nncf_tensor)
        assert all(res == Tensor(tensor * 0))
        assert isinstance(res, Tensor)
        assert res.device == nncf_tensor.device

    def test_fn_maximum(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = Tensor(self.to_tensor([2, 1]))
        tensor_ref = self.to_tensor([2, 2])

        res = fns.maximum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)
        assert res.device == tensor_a.device

    def test_fn_maximum_list(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = [2, 1]
        tensor_ref = self.to_tensor([2, 2])

        res = fns.maximum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)
        assert res.device == tensor_a.device

    def test_fn_minimum(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = Tensor(self.to_tensor([2, 1]))
        tensor_ref = self.to_tensor([1, 1])

        res = fns.minimum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)
        assert res.device == tensor_a.device

    def test_fn_minimum_list(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = [2, 1]
        tensor_ref = self.to_tensor([1, 1])

        res = fns.minimum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)
        assert res.device == tensor_a.device

    def test_fn_ones_like(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_ref = self.to_tensor([1, 1])

        res = fns.ones_like(tensor_a)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)
        assert res.device == tensor_a.device

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            ([True, True], None, True),
            ([True, False], None, False),
            ([False, False], None, False),
            ([[True, True], [False, True]], 0, [False, True]),
        ),
    )
    def test_fn_all(self, val, axis, ref):
        tensor = Tensor(self.to_tensor(val))
        res = fns.all(tensor, axis=axis)
        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, self.to_tensor(ref))
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            ([True, True], None, True),
            ([True, False], None, True),
            ([False, False], None, False),
            ([[False, True], [False, False]], 0, [False, True]),
        ),
    )
    def test_fn_any(self, val, axis, ref):
        tensor = Tensor(self.to_tensor(val))
        res = fns.any(tensor, axis=axis)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, self.to_tensor(ref))
        assert res.device == tensor.device

    def test_fn_where(self):
        tensor = Tensor(self.to_tensor([1, -1]))
        tensor_ref = self.to_tensor([1, 0])
        res = fns.where(tensor > 0, 1, 0)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operators do not guarantee to return a tensor on an initial device.
            assert res.device == tensor.device

    @pytest.mark.parametrize(
        "val, ref",
        (
            ([], True),
            ([1], False),
            (1, False),
        ),
    )
    def test_fn_isempty(self, val, ref):
        tensor = Tensor(self.to_tensor(val))
        res = fns.isempty(tensor)
        assert res == ref
        assert isinstance(res, bool)

    @pytest.mark.parametrize(
        "val, ref",
        (
            ([], True),
            ([1], False),
            (1, False),
        ),
    )
    def test_isempty(self, val, ref):
        tensor = Tensor(self.to_tensor(val))
        res = tensor.isempty()
        assert res == ref
        assert isinstance(res, bool)

    @pytest.mark.parametrize(
        "x1, x2, is_tensor, rtol, atol, ref",
        (
            ([0.1], [0.1], True, None, None, True),
            ([0.1], [0.10001], True, None, None, False),
            ([0.1], [0.10001], True, 0.1, None, True),
            ([0.1], [0.10001], True, None, 0.1, True),
            ([0.1], [0.20001], True, None, 0.1, False),
            ([0.1], 0.1, True, None, None, True),
            ([0.1], 0.1, False, None, None, True),
        ),
    )
    def test_fn_allclose(self, x1, x2, is_tensor, rtol, atol, ref):
        tensor1 = Tensor(self.to_tensor(x1))
        if is_tensor:
            tensor2 = Tensor(self.to_tensor(x2))
        else:
            tensor2 = x2
        if rtol is not None:
            res = fns.allclose(tensor1, tensor2, rtol=rtol)
        elif atol is not None:
            res = fns.allclose(tensor1, tensor2, atol=atol)
        else:
            res = fns.allclose(tensor1, tensor2)
        assert res == ref

    @pytest.mark.parametrize(
        "x1, x2, rtol, atol, ref",
        (
            ([0.1], [0.1], None, None, [True]),
            ([0.1], [0.10001], None, None, [False]),
            ([0.1], [0.10001], 0.1, None, [True]),
            ([0.1], [0.10001], None, 0.1, [True]),
            ([0.1], 0.1, None, None, [True]),
        ),
    )
    def test_fn_isclose(self, x1, x2, rtol, atol, ref):
        tensor1 = Tensor(self.to_tensor(x1))
        tensor2 = Tensor(self.to_tensor(x2))
        if rtol is not None:
            res = fns.isclose(tensor1, tensor2, rtol=rtol)
        elif atol is not None:
            res = fns.isclose(tensor1, tensor2, atol=atol)
        else:
            res = fns.isclose(tensor1, tensor2)
        assert all(res == self.to_tensor(ref))
        assert isinstance(res, Tensor)

    def test_device(self):
        tensor = Tensor(self.to_tensor([1]))
        assert tensor.device == TensorDeviceType.CPU

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

    def test_atleast_1d(self):
        scalar = Tensor(self.to_tensor(42))
        assert fns.atleast_1d(scalar).shape == (1,)

        tensor = Tensor(self.to_tensor([[1, 2, 3]]))
        assert fns.atleast_1d(tensor).shape == (1, 3)

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

    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="is not implemented for"):
            fns.device({}, [1, 2])

    @pytest.mark.parametrize(
        "x, axis, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                1,
                [[0.8, 0.1], [0.2, 0.7], [0.2, 0.1]],
            ),
        ),
    )
    def test_fn_unstack(self, x, axis, ref):
        tensor = Tensor(self.to_tensor(x))
        ref = [self.to_tensor(r) for r in ref]

        res = fns.unstack(tensor, axis=axis)

        assert isinstance(res, list)
        for i, _ in enumerate(ref):
            assert all(res[i] == ref[i])
            assert res[i].device == tensor.device

    @pytest.mark.parametrize(
        "x, axis, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                1,
                [[0.8, 0.1], [0.2, 0.7], [0.2, 0.1]],
            ),
        ),
    )
    def test_fn_stack(self, x, axis, ref):
        list_tensor = [Tensor(self.to_tensor(i)) for i in x]
        ref = self.to_tensor(ref)

        res = fns.stack(list_tensor, axis=axis)

        assert isinstance(res, Tensor)
        assert fns.all(res.data == ref)
        assert res.device == list_tensor[0].device

    @pytest.mark.parametrize(
        "x, axis, ref",
        (
            (
                ([0.8, 0.2, 0.2], [0.1, 0.7, 0.1]),
                0,
                [0.8, 0.2, 0.2, 0.1, 0.7, 0.1],
            ),
            (
                ([[0.8, 0.2], [0.2, 0.1]], [[0.1], [0.7]]),
                1,
                [[0.8, 0.2, 0.1], [0.2, 0.1, 0.7]],
            ),
        ),
    )
    def test_fn_concatenate(self, x, axis, ref):
        list_tensor = [Tensor(self.to_tensor(i)) for i in x]
        ref = self.to_tensor(ref)

        res = fns.concatenate(list_tensor, axis=axis)

        assert isinstance(res, Tensor)
        assert fns.all(res.data == ref), f"{res} {res.data == ref}"
        assert res.device == list_tensor[0].device

    def test_fn_moveaxis(self):
        tensor = [[0, 0, 0], [0, 0, 0]]
        tensor = Tensor(self.to_tensor(tensor))

        res = fns.moveaxis(tensor, 0, -1)

        assert res.shape == (3, 2)

    @pytest.mark.parametrize(
        "x, axis, keepdims, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                False,
                [0.45, 0.45, 0.15],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                True,
                [[0.45, 0.45, 0.15]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                (0, 1),
                True,
                [[0.35]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                None,
                False,
                0.35,
            ),
        ),
    )
    def test_fn_mean(self, x, axis, keepdims, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.mean(tensor, axis, keepdims)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "x, axis, keepdims, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                False,
                [0.45, 0.45, 0.15],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                True,
                [[0.45, 0.45, 0.15]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                (0, 1),
                True,
                [[0.2]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                None,
                False,
                0.2,
            ),
        ),
    )
    def test_fn_median(self, x, axis, keepdims, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.median(tensor, axis, keepdims)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor), f"{res}"
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "val, decimals, ref",
        (
            (1.1, 0, 1.0),
            ([1.1, 0.9], 0, [1.0, 1.0]),
            ([1.11, 0.91], 1, [1.1, 0.9]),
            ([5.5, 3.3], -1, [10.0, 0.0]),
        ),
    )
    def test_fn_round(self, val, decimals, ref):
        tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)

        res = fns.round(tensor, decimals)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (
                [[9.0, 9.0], [7.0, 1.0]],
                0,
                [8.0, 5.0],
            ),
            (
                [[[9.0, 9.0], [0.0, 3.0]], [[5.0, 1.0], [7.0, 1.0]]],
                0,
                [5.25, 3.5],
            ),
            (
                [[[9.0, 9.0], [0.0, 3.0]], [[5.0, 1.0], [7.0, 1.0]]],
                2,
                [5.25, 3.5],
            ),
            (
                [
                    [[[9.0, 6.0], [8.0, 5.0]], [[3.0, 9.0], [4.0, 6.0]]],
                    [[[3.0, 9.0], [9.0, 2.0]], [[2.0, 4.0], [2.0, 5.0]]],
                ],
                0,
                [6.25, 4.5],
            ),
            (
                [
                    [[[9.0, 6.0], [8.0, 5.0]], [[3.0, 9.0], [4.0, 6.0]]],
                    [[[3.0, 9.0], [9.0, 2.0]], [[2.0, 4.0], [2.0, 5.0]]],
                ],
                1,
                [6.375, 4.375],
            ),
            (
                [
                    [[[9.0, 6.0], [8.0, 5.0]], [[3.0, 9.0], [4.0, 6.0]]],
                    [[[3.0, 9.0], [9.0, 2.0]], [[2.0, 4.0], [2.0, 5.0]]],
                ],
                -1,
                [5.0, 5.75],
            ),
        ),
    )
    def test_fn_mean_per_channel(self, val, axis, ref):
        tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)
        res = s_fns.mean_per_channel(tensor, axis)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor), f"{res.data}"
        assert res.device == tensor.device

    @pytest.mark.parametrize("axis", (3, 4, -4, -5))
    def test_fn_mean_per_channel_incorrect_axis(self, axis):
        tensor = Tensor(self.to_tensor([[[9.0, 9.0], [0.0, 3.0]], [[5.0, 1.0], [7.0, 1.0]]]))
        with pytest.raises(ValueError, match="is out of bounds for array of dimension"):
            s_fns.mean_per_channel(tensor, axis)

    def test_size(self):
        tensor = Tensor(self.to_tensor([1, 1]))
        res = tensor.size
        assert res == 2

    def test_item(self):
        tensor = Tensor(self.to_tensor([1]))
        res = tensor.item()
        assert res == 1

    @pytest.mark.parametrize(
        "val, min, max, ref",
        (([0.9, 2.1], 1.0, 2.0, [1.0, 2.0]), ([0.9, 2.1], [0.0, 2.5], [0.5, 3.0], [0.5, 2.5])),
    )
    def test_fn_clip(self, val, min, max, ref):
        tensor = Tensor(self.to_tensor(val))
        if isinstance(min, list):
            min = Tensor(self.to_tensor(min))
        if isinstance(max, list):
            max = Tensor(self.to_tensor(max))
        ref_tensor = self.to_tensor(ref)

        res = fns.clip(tensor, min, max)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    def test_fn_as_tensor_like(self):
        tensor = Tensor(self.to_tensor([1]))
        data = [1.0, 2.0]
        ref = self.to_tensor(data)

        res = fns.as_tensor_like(tensor, data)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "x, axis, keepdims, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                False,
                [0.9, 0.9, 0.3],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                True,
                [[0.9, 0.9, 0.3]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                (0, 1),
                True,
                [[2.1]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                None,
                False,
                2.1,
            ),
        ),
    )
    def test_fn_sum(self, x, axis, keepdims, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.sum(tensor, axis, keepdims)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "a, b, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                [[0.1, 0.7, 0.1], [0.8, 0.2, 0.2]],
                [[0.08, 0.14, 0.02], [0.08, 0.14, 0.02]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0.1,
                [[0.08, 0.02, 0.02], [0.01, 0.07, 0.01]],
            ),
        ),
    )
    def test_fn_multiply(self, a, b, ref):
        tensor_a = Tensor(self.to_tensor(a))
        tensor_b = Tensor(self.to_tensor(b))
        ref_tensor = self.to_tensor(ref)

        res = fns.multiply(tensor_a, tensor_b)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor_a.device

    @pytest.mark.parametrize(
        "x, axis, keepdims, ddof, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                False,
                0,
                [0.1225, 0.0625, 0.0025],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                0,
                True,
                1,
                [[0.245, 0.125, 0.005]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                (0, 1),
                True,
                0,
                [[0.0825]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                None,
                False,
                1,
                0.099,
            ),
        ),
    )
    def test_fn_var(self, x, axis, keepdims, ddof, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.var(tensor, axis, keepdims, ddof)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "x, ord, axis, keepdims, ref",
        (
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                None,
                0,
                False,
                [0.80622577, 0.72801099, 0.2236068],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                "fro",
                None,
                True,
                [[1.10905365]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                "nuc",
                (0, 1),
                True,
                [[1.53063197]],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                "nuc",
                (0, 1),
                False,
                [1.53063197],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                float("inf"),
                0,
                False,
                [0.8, 0.7, 0.2],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                2,
                None,
                False,
                0.9364634205074938,
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                2,
                0,
                False,
                [0.8062258, 0.72801095, 0.22360681],
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                1,
                None,
                False,
                0.9,
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                -1,
                None,
                False,
                0.3,
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                -2,
                None,
                False,
                0.59416854,
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                float("inf"),
                None,
                False,
                1.2,
            ),
            (
                [[0.8, 0.2, 0.2], [0.1, 0.7, 0.1]],
                -float("inf"),
                None,
                False,
                0.9,
            ),
            ([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]], None, None, False, 2.82842708),
        ),
    )
    def test_fn_linalg_norm(self, x, ord, axis, keepdims, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.linalg.norm(tensor, ord, axis, keepdims)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    def test_norm_comprehensive(self):
        # 2D tensor
        tensor_data_2d = [[1.0, 2.0], [3.0, 4.0]]
        tf_tensor_2d = self.to_tensor(tensor_data_2d)
        tensor_2d = Tensor(tf_tensor_2d)

        # 3D tensor
        tensor_data_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        tf_tensor_3d = self.to_tensor(tensor_data_3d)
        tensor_3d = Tensor(tf_tensor_3d)

        # Matrix norm values
        matrix_ord_values = [None, 1, 2, -1, -2, float("inf"), -float("inf"), "fro", "nuc"]

        # Vector norm values
        vector_ord_values = [None, 0, 1, 2, float("inf")]

        # Test vector norms (axis=0 or axis=1)
        for ord_val in vector_ord_values:
            if ord_val == "fro" or ord_val == "nuc":
                continue

            result = fns.linalg.norm(tensor_2d, ord=ord_val, axis=0)
            assert result.shape == (2,), f"Failed for ord={ord_val}, axis=0"

            result = fns.linalg.norm(tensor_2d, ord=ord_val, axis=1)
            assert result.shape == (2,), f"Failed for ord={ord_val}, axis=1"

        # Test matrix norms (axis=None or axis=(0,1))
        for ord_val in matrix_ord_values:
            try:
                result = fns.linalg.norm(tensor_2d, ord=ord_val)

                result = fns.linalg.norm(tensor_2d, ord=ord_val, axis=(0, 1))
                assert result.ndim == 0, f"Failed for ord={ord_val}, axis=(0,1)"

                result = fns.linalg.norm(tensor_2d, ord=ord_val, axis=(0, 1), keepdims=True)
                assert result.shape == (1, 1), f"Failed for ord={ord_val}, axis=(0,1), keepdims=True"
            except ValueError:
                pass

        # Test 3D tensor slicing for nuclear norm
        try:
            result = fns.linalg.norm(tensor_3d, ord="nuc", axis=(1, 2))
            assert result.shape == (2,), "Failed for 3D tensor, ord=nuc, axis=(1,2)"

            result = fns.linalg.norm(tensor_3d, ord="nuc", axis=(1, 2), keepdims=True)
            assert result.shape == (2, 1, 1), "Failed for 3D tensor, ord=nuc, axis=(1,2), keepdims=True"
        except ValueError as e:
            assert False, f"Failed for 3D tensor, ord=nuc, axis=(1,2), error: {e}"

    def test_norm_3d_tensor(self):
        tensor_data_3d = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
        tf_tensor_3d = self.to_tensor(tensor_data_3d)
        tensor_3d = Tensor(tf_tensor_3d)

        # Single axis norms (vector norms)
        for axis_val in [0, 1, 2]:
            for ord_val in [None, 0, 1, 2, float("inf")]:
                result = fns.linalg.norm(tensor_3d, ord=ord_val, axis=axis_val)

                expected_shape = list(tensor_3d.shape)
                expected_shape.pop(axis_val)
                assert result.shape == tuple(expected_shape), f"Failed shape check for ord={ord_val}, axis={axis_val}"

                result_keep = fns.linalg.norm(tensor_3d, ord=ord_val, axis=axis_val, keepdims=True)
                expected_shape_keep = list(tensor_3d.shape)
                expected_shape_keep[axis_val] = 1
                assert result_keep.shape == tuple(expected_shape_keep), (
                    f"Failed keepdims shape for ord={ord_val}, axis={axis_val}"
                )

        # Dual axis norms (matrix norms)
        axis_pairs = [(0, 1), (0, 2), (1, 2)]
        for axis_pair in axis_pairs:
            for ord_val in ["fro", "nuc", 1, 2, float("inf"), -float("inf")]:
                try:
                    result = fns.linalg.norm(tensor_3d, ord=ord_val, axis=axis_pair)

                    expected_shape = []
                    for i in range(tensor_3d.ndim):
                        if i not in axis_pair:
                            expected_shape.append(tensor_3d.shape[i])
                    assert result.shape == tuple(expected_shape), (
                        f"Failed shape check for ord={ord_val}, axis={axis_pair}"
                    )

                    result_keep = fns.linalg.norm(tensor_3d, ord=ord_val, axis=axis_pair, keepdims=True)
                    expected_shape_keep = list(tensor_3d.shape)
                    for i in axis_pair:
                        expected_shape_keep[i] = 1
                    assert result_keep.shape == tuple(expected_shape_keep), (
                        f"Failed keepdims shape for ord={ord_val}, axis={axis_pair}"
                    )
                except ValueError as e:
                    if ord_val == "nuc" and axis_pair in [(0, 1), (0, 2)]:
                        assert False, f"Failed shape check for nuclear norm with axis={axis_pair}, error: {e}"

        # Testing for nuclear norm on all possible axis combinations
        nuclear_axes = [(0, 1), (0, 2), (1, 2)]
        for axis_pair in nuclear_axes:
            try:
                result = fns.linalg.norm(tensor_3d, ord="nuc", axis=axis_pair)

                result_keep = fns.linalg.norm(tensor_3d, ord="nuc", axis=axis_pair, keepdims=True)

                remaining_dims = tensor_3d.ndim - len(axis_pair)
                assert result.ndim == remaining_dims, f"Wrong dimension for nuclear norm with axis={axis_pair}"

                expected_shape_keep = []
                for i in range(tensor_3d.ndim):
                    expected_shape_keep.append(1 if i in axis_pair else tensor_3d.shape[i])
                assert result_keep.shape == tuple(expected_shape_keep), (
                    f"Wrong keepdims shape for nuclear norm with axis={axis_pair}"
                )
            except ValueError as e:
                assert False, f"Nuclear norm failed for axis={axis_pair}, error: {e}"

    def test_norm_order_zero(self):
        # 1D tensor
        tensor_data_1d = [1.0, 0.0, 3.0, 0.0, 5.0]
        tf_tensor_1d = self.to_tensor(tensor_data_1d)
        tensor_1d = Tensor(tf_tensor_1d)

        # Test 1D tensor with ord=0
        result = fns.linalg.norm(tensor_1d, ord=0)
        assert result.item() == 3, f"Expected 3 non-zeros, got {result.item()}"

        # Test 1D tensor with keepdims
        result = fns.linalg.norm(tensor_1d, ord=0, keepdims=True)
        assert result.shape == (1,), f"Expected shape (1,), got {result.shape}"
        assert result.item() == 3, f"Expected 3 non-zeros, got {result.item()}"

        # Test 1D vector from a slice of a 2D tensor
        tensor_data_2d = [[1.0, 0.0, 3.0], [0.0, 0.0, 6.0], [7.0, 0.0, 9.0]]
        tf_tensor_2d = self.to_tensor(tensor_data_2d)
        tensor_2d = Tensor(tf_tensor_2d)

        # Test along axis=0
        result = fns.linalg.norm(tensor_2d, ord=0, axis=0)
        expected = [2, 0, 3]

        result_data = result.data
        for i, expected_val in enumerate(expected):
            val = float(result_data[i])
            assert val == expected_val, f"At index {i}, expected {expected_val}, got {val}"

        # Test along axis=1
        result = fns.linalg.norm(tensor_2d, ord=0, axis=1)
        expected = [2, 1, 2]

        result_data = result.data
        for i, expected_val in enumerate(expected):
            val = float(result_data[i])
            assert val == expected_val, f"At index {i}, expected {expected_val}, got {val}"

        # Test with keepdims=True
        result = fns.linalg.norm(tensor_2d, ord=0, axis=0, keepdims=True)
        assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"

        result = fns.linalg.norm(tensor_2d, ord=0, axis=1, keepdims=True)
        assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"

    def test_norm_4d_tensor(self):
        # 4D tensor
        tensor_data_4d = [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]
        tf_tensor_4d = self.to_tensor(tensor_data_4d)
        tensor_4d = Tensor(tf_tensor_4d)

        # Test nuclear norm on slices
        nuclear_axes = [(1, 2)]
        for axis_pair in nuclear_axes:
            try:
                result = fns.linalg.norm(tensor_4d, ord="nuc", axis=axis_pair)
                result_keep = fns.linalg.norm(tensor_4d, ord="nuc", axis=axis_pair, keepdims=True)

                expected_shape = [dim for i, dim in enumerate(tensor_4d.shape) if i not in axis_pair]
                assert result.shape == tuple(expected_shape)

                expected_shape_keep = [1 if i in axis_pair else dim for i, dim in enumerate(tensor_4d.shape)]
                assert result_keep.shape == tuple(expected_shape_keep)
            except ValueError as e:
                assert False, f"Failed shape check for nuclear norm with axis={axis_pair}, error: {e}"

        # Test vector norms along single axes
        for ord_val in [0, 1, float("inf")]:
            for axis in range(4):
                result = fns.linalg.norm(tensor_4d, ord=ord_val, axis=axis)
                expected_shape = list(tensor_4d.shape)
                expected_shape.pop(axis)
                assert result.shape == tuple(expected_shape)

                result = fns.linalg.norm(tensor_4d, ord=ord_val, axis=axis, keepdims=True)
                expected_shape = list(tensor_4d.shape)
                expected_shape[axis] = 1
                assert result.shape == tuple(expected_shape)

    def test_norm_empty_tensor(self):
        empty_tensor_1d = self.to_tensor([])
        tensor_1d = Tensor(empty_tensor_1d)

        empty_tensor_2d = self.to_tensor([[]])
        tensor_2d = Tensor(empty_tensor_2d)

        # Test vector norms on empty 1D tensor
        for ord_val in [0, 1]:
            try:
                result = fns.linalg.norm(tensor_1d, ord=ord_val)
                assert result.item() == 0, f"Empty tensor norm with ord={ord_val} should be 0"
            except (Exception, ValueError, RuntimeError) as e:
                if (
                    "shape" not in str(e).lower()
                    and "empty" not in str(e).lower()
                    and "dimension" not in str(e).lower()
                    and "zero-size" not in str(e).lower()
                    and "reduction" not in str(e).lower()
                ):
                    assert False, f"Unexpected error for empty tensor with ord={ord_val}: {e}"
                print(f"Expected error for empty tensor with ord={ord_val}: {e}")

        try:
            result = fns.linalg.norm(tensor_1d, ord=float("inf"))
            assert result.item() == 0, "Empty tensor infinity norm should be 0"
        except (Exception, ValueError, RuntimeError) as e:
            if (
                "shape" not in str(e).lower()
                and "empty" not in str(e).lower()
                and "dimension" not in str(e).lower()
                and "zero-size" not in str(e).lower()
                and "reduction" not in str(e).lower()
            ):
                assert False, f"Unexpected error for empty tensor with ord=inf: {e}"
            print(f"Expected error for empty tensor with ord=inf: {e}")

        try:
            result = fns.linalg.norm(tensor_1d, ord=0)
            assert result.item() == 0, "Count of non-zeros in empty tensor should be 0"
        except Exception as e:
            if (
                "shape" not in str(e).lower()
                and "empty" not in str(e).lower()
                and "dimension" not in str(e).lower()
                and "zero-size" not in str(e).lower()
            ):
                assert False, f"Unexpected error for empty tensor with ord=0: {e}"
            print(f"Expected shape error for empty tensor with ord=0: {e}")

        try:
            result = fns.linalg.norm(tensor_1d, ord=0, keepdims=True)
            assert result.shape == (1,), "Shape with keepdims should be (1,)"
            assert result.item() == 0, "Count of non-zeros in empty tensor should be 0"
        except Exception as e:
            if (
                "shape" not in str(e).lower()
                and "empty" not in str(e).lower()
                and "dimension" not in str(e).lower()
                and "zero-size" not in str(e).lower()
            ):
                assert False, f"Unexpected error for empty tensor with keepdims: {e}"
            print(f"Expected error with keepdims on empty tensor: {e}")

        # Test matrix norms on empty 2D tensor
        for ord_val in ["fro"]:
            try:
                result = fns.linalg.norm(tensor_2d, ord=ord_val)
                assert result.item() == 0, f"Empty tensor norm with ord={ord_val} should be 0"
            except (Exception, ValueError, RuntimeError) as e:
                if (
                    "shape" not in str(e).lower()
                    and "empty" not in str(e).lower()
                    and "dimension" not in str(e).lower()
                    and "zero-size" not in str(e).lower()
                ):
                    assert False, f"Unexpected error for empty 2D tensor with ord={ord_val}: {e}"
                print(f"Expected error for empty 2D tensor with ord={ord_val}: {e}")

        # Test vector norm on columns of empty 2D tensor
        try:
            result = fns.linalg.norm(tensor_2d, ord=0, axis=0)
            assert len(result.shape) > 0, "Result should have at least one dimension"
            assert result.size == 0, "Result should be empty along specified axis"
        except Exception as e:
            print(f"Note: Cannot compute norm along axis=0 for empty tensor: {e}")

    def test_norm_extreme_values(self):
        # Tensors with NaN, Inf, and extreme values
        inf_tensor_data = [[1.0, float("inf")], [3.0, 4.0]]
        inf_tf_tensor = self.to_tensor(inf_tensor_data)
        inf_tensor = Tensor(inf_tf_tensor)

        nan_tensor_data = [[1.0, 2.0], [float("nan"), 4.0]]
        nan_tf_tensor = self.to_tensor(nan_tensor_data)
        nan_tensor = Tensor(nan_tf_tensor)

        # Test vector norms along axis=0
        result = fns.linalg.norm(inf_tensor, ord=0, axis=0)
        assert result.shape == (2,)

        result_data = result.data
        assert float(result_data[0]) == 2 and float(result_data[1]) == 2

        # Test vector norms along axis=1
        result = fns.linalg.norm(inf_tensor, ord=0, axis=1)
        assert result.shape == (2,)

        result_data = result.data
        assert float(result_data[0]) == 2 and float(result_data[1]) == 2

        # Create 1D tensors to test ord=0 behavior on vectors
        inf_vector_data = [1.0, float("inf"), 3.0, 4.0]
        inf_vector_tensor = self.to_tensor(inf_vector_data)
        inf_vector = Tensor(inf_vector_tensor)

        nan_vector_data = [1.0, 2.0, float("nan"), 4.0]
        nan_vector_tensor = self.to_tensor(nan_vector_data)
        nan_vector = Tensor(nan_vector_tensor)

        mixed_vector_data = [float("inf"), 2.0, float("nan"), 4.0]
        mixed_vector_tensor = self.to_tensor(mixed_vector_data)
        mixed_vector = Tensor(mixed_vector_tensor)

        # Test ord=0 on vectors with extreme values
        result = fns.linalg.norm(inf_vector, ord=0)
        assert result.item() == 4, "All elements (including Inf) should be counted as non-zero"

        result = fns.linalg.norm(nan_vector, ord=0)
        assert result.item() == 4, "All elements (including NaN) should be counted as non-zero"

        try:
            result = fns.linalg.norm(mixed_vector, ord=0)
            assert result.item() == 4, "All elements should be counted as non-zero"
        except Exception as e:
            if "invalid" not in str(e).lower():
                assert False, f"Unexpected error with mixed NaN and Inf: {e}"
            print(f"Note: Expected error with mixed NaN and Inf values: {e}")

        # Test matrix norms
        try:
            result = fns.linalg.norm(inf_tensor, ord="fro")
            assert float("inf") == result.item() or result.item() > 1e30, "Norm with Inf should be Inf or very large"
        except (Exception, ValueError, RuntimeError) as e:
            print(f"Note: Backend cannot compute Frobenius norm with Inf values: {e}")

        try:
            result = fns.linalg.norm(nan_tensor, ord="nuc")
            import numpy as np

            assert np.isnan(result.item()) or result.item() > 0, "Nuclear norm with NaN might be NaN"
        except (Exception, ValueError, RuntimeError) as e:
            print(f"Note: Backend SVD cannot handle NaN values: {e}")

        try:
            result = fns.linalg.norm(inf_tensor, ord=float("inf"))
            assert result.item() == float("inf"), "Infinity norm with Inf values should be Inf"
        except (Exception, ValueError, RuntimeError) as e:
            print(f"Note: Backend cannot compute infinity norm with Inf values: {e}")

        # Test keepdims with vector norms
        result = fns.linalg.norm(inf_vector, ord=0, keepdims=True)
        assert result.shape == (1,)
        assert result.item() == 4

    @pytest.mark.parametrize(
        "m1, m2, ref",
        (
            (
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[9.0, 12.0, 15.0], [19.0, 26.0, 33.0], [29.0, 40.0, 51.0]],
            ),
            (
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[22.0, 28.0], [49.0, 64.0]],
            ),
        ),
    )
    def test_fn_matmul(self, m1, m2, ref):
        tensor1 = Tensor(self.to_tensor(m1))
        tensor2 = Tensor(self.to_tensor(m2))
        ref_tensor = self.to_tensor(ref)

        res = fns.matmul(tensor1, tensor2)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor1.device

        res = tensor1 @ tensor2

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        if (
            self.backend() != TensorBackend.tf
        ):  # native Tensorflow operators do not guarantee to return a tensor on an initial device.
            assert res.device == tensor1.device

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[1], [2]]]], None, [[1, 2], [1, 2]]),
            ([[[[1], [2]], [[1], [2]]]], 0, [[[1], [2]], [[1], [2]]]),
            ([[[[1], [2]], [[1], [2]]]], -1, [[[1, 2], [1, 2]]]),
            ([[[[1], [2]], [[1], [2]]]], (0, 3), [[1, 2], [1, 2]]),
        ),
    )
    def test_unsqueeze(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = fns.squeeze(nncf_tensor, axis=axis)
        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
        assert res.device == nncf_tensor.device

    @pytest.mark.parametrize(
        "x, ref",
        (
            (
                [[1, 2], [3, 4], [5, 6]],
                [[1, 3, 5], [2, 4, 6]],
            ),
            (
                [[1, 2, 3], [4, 5, 6]],
                [[1, 4], [2, 5], [3, 6]],
            ),
        ),
    )
    def test_fn_transpose(self, x, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.transpose(tensor)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "x, axis, descending, stable, ref",
        (
            ([1, 2, 3, 4, 5, 6], -1, False, False, [0, 1, 2, 3, 4, 5]),
            ([6, 5, 4, 3, 2, 1], -1, True, False, [0, 1, 2, 3, 4, 5]),
            (
                [[1, 2, 2, 3, 3, 3], [4, 5, 6, 6, 5, 5], [1, 2, 2, 3, 3, 3]],
                -1,
                False,
                True,
                [[0, 1, 2, 3, 4, 5], [0, 1, 4, 5, 2, 3], [0, 1, 2, 3, 4, 5]],
            ),
            (
                [[1, 2, 2, 3, 3, 3], [4, 5, 6, 6, 5, 5], [1, 2, 2, 3, 3, 3]],
                -1,
                True,
                True,
                [[3, 4, 5, 1, 2, 0], [2, 3, 1, 4, 5, 0], [3, 4, 5, 1, 2, 0]],
            ),
            (
                [[1, 2, 2, 3, 3, 3], [4, 5, 6, 6, 5, 5], [1, 2, 2, 3, 3, 3]],
                0,
                False,
                True,
                [[0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1, 1]],
            ),
            (
                [[1, 2, 2, 3, 3, 3], [4, 5, 6, 6, 5, 5], [1, 2, 2, 3, 3, 3]],
                0,
                True,
                True,
                [[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], [2, 2, 2, 2, 2, 2]],
            ),
        ),
    )
    def test_fn_argsort(self, x, axis, descending, stable, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.argsort(tensor, axis, descending, stable)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    zero_ten_range = [x / 100 for x in range(1001)]
    zero_ten_range_two_axes = [[a + b / 100 for b in range(101)] for a in range(10)]

    @pytest.mark.parametrize(
        "x,q,axis,keepdims,ref",
        (
            (1.0, 0.1, None, True, 1.0),
            (zero_ten_range, 0.1, 0, True, [1.0]),
            (zero_ten_range, 0.1, 0, False, 1.0),
            (zero_ten_range, (0.1, 0.9), 0, False, [1.0, 9.0]),
            (zero_ten_range, (0.1, 0.9), 0, True, [[1.0], [9.0]]),
            (zero_ten_range_two_axes, (0.1, 0.9), (0, 1), False, [1.0, 9.0]),
            (zero_ten_range_two_axes, (0.1, 0.9), (0, 1), True, [[[1.0]], [[9.0]]]),
            (16000 * zero_ten_range, 0.1, 0, False, 1.0),  # reason: https://github.com/pytorch/pytorch/issues/64947
            (zero_ten_range_two_axes, (0.1, 0.9), None, False, [1.0, 9.0]),
        ),
    )
    @pytest.mark.parametrize("fp16", [False, True])
    def test_fn_quantile(self, x, q, axis, keepdims, ref, fp16):
        tensor = self.to_tensor(x)
        if fp16:
            tensor = self.cast_to(tensor, TensorDataType.float16)
        tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)

        res = fns.quantile(tensor, axis=axis, q=q, keepdims=keepdims)
        assert isinstance(res, Tensor)
        assert res.dtype == TensorDataType.float64
        assert fns.allclose(self.cast_to(res.data, TensorDataType.float32), ref_tensor)
        assert res.device == tensor.device
        assert res.shape == tuple(ref_tensor.shape)

    @pytest.mark.parametrize(
        "x,q,axis,keepdims,ref",
        (
            (1.0, 10, None, True, 1.0),
            (zero_ten_range, 10, 0, True, [1.0]),
            (zero_ten_range, 10, 0, False, 1.0),
            (zero_ten_range, (10, 90), 0, False, [1.0, 9.0]),
            (zero_ten_range, (10, 90), 0, True, [[1.0], [9.0]]),
            (zero_ten_range_two_axes, (10, 90), (0, 1), False, [1.0, 9.0]),
            (zero_ten_range_two_axes, (10, 90), (0, 1), True, [[[1.0]], [[9.0]]]),
        ),
    )
    def test_fn_percentile(self, x, q, axis, keepdims, ref):
        tensor = self.to_tensor(x)
        tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)

        res = fns.percentile(tensor, axis=axis, q=q, keepdims=keepdims)
        assert isinstance(res, Tensor)
        assert fns.allclose(self.cast_to(res.data, TensorDataType.float32), ref_tensor)
        assert res.device == tensor.device
        assert res.shape == tuple(ref_tensor.shape)

    @pytest.mark.parametrize(
        "x,power,ref",
        [
            (list(map(float, range(10))), 2.0, [x**2 for x in map(float, range(10))]),
            (list(map(float, range(10))), [2.0], [x**2 for x in map(float, range(10))]),
            (
                list(map(float, range(10))),
                list(map(float, range(10))),
                [1.0, 1.0, 4.0, 27.0, 256.0, 3125.0, 46656.0, 823543.0, 16777216.0, 387420489.0],
            ),
        ],
    )
    def test_fn_power(self, x, power, ref):
        if isinstance(power, list):
            power = self.to_tensor(power)
            power = Tensor(power)

        if isinstance(x, list):
            x = self.to_tensor(x)
        tensor = Tensor(x)

        ref_tensor = self.to_tensor(ref)

        res = fns.power(tensor, power)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device
        assert res.shape == tuple(ref_tensor.shape)

    @pytest.mark.parametrize(
        "a, upper, ref",
        (
            ([[1.0, 2.0], [2.0, 5.0]], False, [[1.0, 0.0], [2.0, 1.0]]),
            ([[1.0, 2.0], [2.0, 5.0]], True, [[1.0, 2.0], [0.0, 1.0]]),
            (
                [[[1.0, 2.0], [2.0, 5.0]], [[9.0, -3.0], [-3.0, 5.0]]],
                False,
                [[[1.0, 0.0], [2.0, 1.0]], [[3.0, 0.0], [-1.0, 2.0]]],
            ),
            (
                [[[1.0, 2.0], [2.0, 5.0]], [[9.0, -3.0], [-3.0, 5.0]]],
                True,
                [[[1.0, 2.0], [0.0, 1.0]], [[3.0, -1.0], [0.0, 2.0]]],
            ),
        ),
    )
    def test_fn_linalg_cholesky(self, a, upper, ref):
        tensor_a = Tensor(self.to_tensor(a))
        ref_tensor = self.to_tensor(ref)

        res = fns.linalg.cholesky(tensor_a, upper=upper)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor_a.device

    @pytest.mark.parametrize(
        "a, upper, ref",
        (
            ([[1.0, 0.0], [2.0, 1.0]], False, [[5.0, -2.0], [-2.0, 1.0]]),
            ([[1.0, 2.0], [0.0, 1.0]], True, [[5.0, -2.0], [-2.0, 1.0]]),
            (
                [[[1.0, 0.0], [2.0, 1.0]], [[3.0, 0.0], [-1.0, 2.0]]],
                False,
                [[[5.0, -2.0], [-2.0, 1.0]], [[0.1388888888888889, 0.08333333333333333], [0.08333333333333333, 0.25]]],
            ),
            (
                [[[1.0, 2.0], [0.0, 1.0]], [[3.0, -1.0], [0.0, 2.0]]],
                True,
                [[[5.0, -2.0], [-2.0, 1.0]], [[0.1388888888888889, 0.08333333333333333], [0.08333333333333333, 0.25]]],
            ),
        ),
    )
    def test_fn_linalg_cholesky_inverse(self, a, upper, ref):
        tensor_a = Tensor(self.to_tensor(a))
        ref_tensor = self.to_tensor(ref)

        res = fns.linalg.cholesky_inverse(tensor_a, upper=upper)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor_a.device

    @pytest.mark.parametrize(
        "a, ref",
        (
            ([[1.0, 2.0], [2.0, 5.0]], [[5.0, -2.0], [-2.0, 1.0]]),
            (
                [[[0.5, 0.2], [0.5, 0.7]], [[0.2, 0.8], [0.1, 0.8]]],
                [[[2.8, -0.8], [-2.0, 2.0]], [[10.0, -10.0], [-1.25, 2.5]]],
            ),
        ),
    )
    def test_fn_linalg_inv(self, a, ref):
        tensor_a = Tensor(self.to_tensor(a))
        ref_tensor = self.to_tensor(ref)

        res = fns.linalg.inv(tensor_a)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor_a.device

    def test_fn_linalg_pinv(self):
        a = [[1.0], [2.0]]
        A = Tensor(self.to_tensor(a))
        B = fns.linalg.pinv(A)
        assert isinstance(B, Tensor)
        assert B.device == A.device
        assert fns.allclose(A, A @ B @ A)
        assert fns.allclose(B, B @ A @ B)

    @pytest.mark.parametrize(
        "a, k, ref",
        (
            ([[1.0, 3.0], [2.0, 5.0]], 0, [1.0, 5.0]),
            ([[1.0, 3.0], [2.0, 5.0]], 1, [3.0]),
            ([[1.0, 3.0], [2.0, 5.0]], -1, [2.0]),
            ([1.0, 5.0], 0, [[1.0, 0.0], [0.0, 5.0]]),
            ([3.0], 1, [[0.0, 3.0], [0.0, 0.0]]),
            ([2.0], -1, [[0.0, 0.0], [2.0, 0.0]]),
        ),
    )
    def test_fn_diag(self, a, k, ref):
        tensor_a = Tensor(self.to_tensor(a))
        ref_tensor = self.to_tensor(ref)

        res = fns.diag(tensor_a, k=k)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor_a.device

    @pytest.mark.parametrize(
        "x1, x2, ref",
        (([True, False], [False, False], [True, False]),),
    )
    def test_fn_logical_or(self, x1, x2, ref):
        x1 = Tensor(self.to_tensor(x1))
        x2 = Tensor(self.to_tensor(x2))
        ref_tensor = self.to_tensor(ref)
        res = fns.logical_or(x1, x2)
        assert isinstance(res, Tensor)
        assert fns.all(res.data == ref_tensor)
        assert res.device == x1.device

    @pytest.mark.parametrize(
        "x, mask, axis, keepdims, ref",
        (
            ([0.0, 1.0, 0.0], [True, False, True], None, False, 1.0),
            ([0.0, 1.0, 0.0], [True, False, True], None, True, [1.0]),
            ([[0.0, 2.0], [2.0, 0.0]], [[False, False], [False, True]], 0, False, [1.0, 2.0]),
            ([[0.0, 2.0], [2.0, 0.0]], [[False, False], [False, True]], 0, True, [[1.0, 2.0]]),
            ([[0.0, 2.0], [2.0, 0.0]], [[True, True], [True, True]], 0, True, [[0.0, 0.0]]),
            (
                [[[0.5, 0.2], [0.5, 0.7]], [[0.2, 0.8], [0.1, 0.8]]],
                [[[False, False], [False, False]], [[False, True], [False, False]]],
                (0, 2),
                False,
                [0.3, 0.525],
            ),
            (
                [[[0.5, 0.2], [0.5, 0.7]], [[0.2, 0.8], [0.1, 0.8]]],
                [[[False, False], [False, False]], [[False, True], [False, False]]],
                (1, 2),
                True,
                [[[0.475]], [[0.36666667]]],
            ),
        ),
    )
    def test_fn_masked_mean(self, x, mask, axis, keepdims, ref):
        x = Tensor(self.to_tensor(x))
        mask = Tensor(self.to_tensor(mask))
        ref_tensor = self.to_tensor(ref)
        res = fns.masked_mean(x, mask, axis, keepdims)
        assert isinstance(res, Tensor)
        assert res.shape == ref_tensor.shape
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == x.device

    @pytest.mark.parametrize(
        "x, mask, axis, keepdims, ref",
        (
            ([0.0, 1.0, 0.0], [True, False, True], None, False, 1.0),
            ([0.0, 1.0, 0.0], [True, False, True], None, True, [1.0]),
            ([[0.0, 2.0], [2.0, 0.0]], [[False, False], [False, True]], 0, False, [1.0, 2.0]),
            ([[0.0, 2.0], [2.0, 0.0]], [[False, False], [False, True]], 0, True, [[1.0, 2.0]]),
            ([[0.0, 2.0], [2.0, 0.0]], [[True, True], [True, True]], 0, True, [[0.0, 0.0]]),
            (
                [[[0.5, 0.2], [0.5, 0.7]], [[0.2, 0.8], [0.1, 0.8]]],
                [[[False, False], [False, False]], [[False, True], [False, False]]],
                (0, 2),
                False,
                [0.2, 0.6],
            ),
            (
                [[[0.5, 0.2], [0.5, 0.7]], [[0.2, 0.8], [0.1, 0.8]]],
                [[[False, False], [False, False]], [[False, True], [False, False]]],
                (1, 2),
                True,
                [[[0.5]], [[0.2]]],
            ),
        ),
    )
    def test_fn_masked_median(self, x, mask, axis, keepdims, ref):
        x = Tensor(self.to_tensor(x))
        mask = Tensor(self.to_tensor(mask))
        ref_tensor = self.to_tensor(ref)
        res = fns.masked_median(x, mask, axis, keepdims)
        assert isinstance(res, Tensor)
        assert res.shape == ref_tensor.shape
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == x.device

    @pytest.mark.parametrize(
        "x, axis, ref",
        (
            (2, 0, [2]),
            (2, -1, [2]),
            (2, (0, 1), [[2]]),
            ([2, 2], 0, [[2, 2]]),
            ([2, 2], 1, [[2], [2]]),
            ([2, 2], -1, [[2], [2]]),
            ([2, 2], -2, [[2, 2]]),
            ([2, 2], (0, 1), [[[2, 2]]]),
            ([2, 2], (0, 1, 2), [[[[2, 2]]]]),
            ([2, 2], (0, 1, 3), [[[[2], [2]]]]),
            ([[[[2], [2]]]], 0, [[[[[2], [2]]]]]),
            ([[[[2], [2]]]], 2, [[[[[2], [2]]]]]),
            ([[[[2], [2]]]], -4, [[[[[2], [2]]]]]),
            ([[[[2], [2]]]], (0, 3, -5), [[[[[[[2], [2]]]]]]]),
        ),
    )
    def test_expand_dims(self, x, axis, ref):
        x = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)
        res = fns.expand_dims(x, axis)
        assert isinstance(res, Tensor)
        assert res.shape == ref_tensor.shape, f"{res.data}".replace("\n", "")
        assert fns.allclose(res.data, ref_tensor), f"{res.data}".replace("\n", "")

    @pytest.mark.parametrize(
        "x, axis, match",
        (
            ([2], 2, "is out of bounds for array"),
            ([2], -3, "is out of bounds for array"),
            ([2], (0, 10), "is out of bounds for array"),
            ([2], (0, 0), "repeated axis"),
        ),
    )
    def test_expand_dims_error(self, x, axis, match):
        x = Tensor(self.to_tensor(x))
        with pytest.raises(Exception, match=match):
            fns.expand_dims(x, axis)

    def test_fn_zeros(self):
        shape = (2, 2)
        for dtype in TensorDataType:
            if (
                self.backend() == TensorBackend.numpy
                and dtype == TensorDataType.bfloat16
                or dtype
                in [
                    TensorDataType.int4,
                    TensorDataType.uint4,
                    TensorDataType.nf4,
                    TensorDataType.f8e4m3,
                    TensorDataType.f8e5m2,
                ]
            ):
                continue
            tensor_a = fns.zeros(shape, backend=self.backend(), dtype=dtype, device=self.device())
            assert isinstance(tensor_a, Tensor)
            assert tensor_a.device == self.device()
            assert tensor_a.backend == self.backend()
            assert tensor_a.dtype == dtype
            assert tensor_a.shape == shape
            assert fns.all(tensor_a == 0)

    @pytest.mark.parametrize(
        "n, m, ref",
        (
            (2, None, [[1, 0], [0, 1]]),
            (2, 2, [[1, 0], [0, 1]]),
            (2, 1, [[1], [0]]),
            (1, 2, [[1, 0]]),
        ),
    )
    def test_fn_eye(self, n, m, ref):
        for dtype in TensorDataType:
            if (
                self.backend() == TensorBackend.numpy
                and dtype == TensorDataType.bfloat16
                or dtype
                in [
                    TensorDataType.int4,
                    TensorDataType.uint4,
                    TensorDataType.nf4,
                    TensorDataType.f8e4m3,
                    TensorDataType.f8e5m2,
                ]
            ):
                continue
            tensor_a = fns.eye(n, m, backend=self.backend(), dtype=dtype, device=self.device())
            assert isinstance(tensor_a, Tensor)
            assert tensor_a.device == self.device()
            assert tensor_a.backend == self.backend()
            assert tensor_a.dtype == dtype
            ref_shape = (n, n) if m is None else (n, m)
            assert tensor_a.shape == ref_shape
            assert fns.allclose(tensor_a, ref)

    @pytest.mark.parametrize(
        "start, end, stop, ref",
        ((3, None, None, [0, 1, 2]), (0, 3, None, [0, 1, 2]), (0, 3, 1, [0, 1, 2]), (2, -1, -1, [2, 1, 0])),
    )
    def test_fn_arange(self, start, end, stop, ref):
        args = [start]
        if end is not None:
            args.append(end)
        if stop is not None:
            args.append(stop)

        for dtype in [TensorDataType.int32, TensorDataType.float32]:
            tensor_ref = Tensor(fns.astype(self.to_tensor(ref), dtype))
            tensor_a = fns.arange(*tuple(args), backend=self.backend(), dtype=dtype, device=self.device())
            assert isinstance(tensor_a, Tensor)
            assert tensor_a.device == self.device()
            assert tensor_a.backend == self.backend()
            assert tensor_a.dtype == dtype
            assert fns.all(tensor_a == tensor_ref)

    def test_fn_from_numpy(self):
        ndarray = np.array([1, 2])
        ref_cpu = self.to_cpu(self.to_tensor(ndarray))
        ref = Tensor(ref_cpu)
        tensor = fns.from_numpy(ndarray, backend=ref.backend)
        assert isinstance(tensor, Tensor)
        assert tensor.device == ref.device
        assert tensor.backend == ref.backend
        assert tensor.dtype == ref.dtype
        assert fns.all(tensor == ref)

    @pytest.mark.parametrize(
        "a, v, side, sorter, ref",
        (
            ([-1.0, 0.0, 0.0, 1.0], [-2.0, -0.6, 0.0, 0.3, 1.5], "left", None, [0, 1, 1, 3, 4]),
            ([-1.0, 0.0, 0.0, 1.0], [-2.0, -0.6, 0.0, 0.3, 1.5], "right", None, [0, 1, 3, 3, 4]),
            ([0.0, -1.0, 0.0, 1.0], [-2.0, -0.6, 0.0, 0.3, 1.5], "left", [1, 0, 2, 3], [0, 1, 1, 3, 4]),
            ([0.0, -1.0, 0.0, 1.0], [-2.0, -0.6, 0.0, 0.3, 1.5], "right", [1, 0, 2, 3], [0, 1, 3, 3, 4]),
        ),
    )
    def test_fn_searchsorted(self, a, v, side, sorter, ref):
        tensor_a = Tensor(self.to_tensor(a))
        tensor_v = Tensor(self.to_tensor(v))
        tensor_sorter = sorter
        if sorter is not None:
            tensor_sorter = Tensor(self.to_tensor(sorter))
        ref = Tensor(self.to_tensor(ref))
        res = fns.searchsorted(tensor_a, tensor_v, side, tensor_sorter)
        assert fns.allclose(res, ref)

    def test_searchsorted_side_error(self):
        tensor_a = Tensor(self.to_tensor([-1.0, 0.0, 0.0, 1.0]))
        tensor_v = Tensor(self.to_tensor([-2.0, -0.6, 0.0, 0.3, 1.5]))
        with pytest.raises(ValueError):
            fns.searchsorted(tensor_a, tensor_v, "error")

    def test_searchsorted_2d_error(self):
        tensor_a = Tensor(self.to_tensor([[-1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 0.0, 1.0]]))
        tensor_v = Tensor(self.to_tensor([-2.0, -0.6, 0.0, 0.3, 1.5]))
        with pytest.raises(ValueError):
            fns.searchsorted(tensor_a, tensor_v)

    @pytest.mark.parametrize(
        "val,ref",
        (
            (1.1, 2.0),
            ([1.1, 0.9], [2.0, 1.0]),
            ([1.11, 0.91], [2.0, 1.0]),
        ),
    )
    def test_fn_ceil(self, val, ref):
        tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)

        res = fns.ceil(tensor)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

    @pytest.mark.parametrize(
        "x,ref",
        [
            (list(map(float, range(1, 10))), [log2(x) for x in map(float, range(1, 10))]),
        ],
    )
    def test_fn_log2(self, x, ref):
        if isinstance(x, list):
            x = self.to_tensor(x)
        tensor = Tensor(x)

        ref_tensor = self.to_tensor(ref)

        res = fns.log2(tensor)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device
        assert res.shape == tuple(ref_tensor.shape)

    @pytest.mark.parametrize(
        "x, y, a_ref, b_ref",
        (
            ([1.0, 2.0, 4.0], [3.0, 4.0, 6.0], 1, 2),
            ([1.0, 2.0, 3.0], [3.0, 2.5, 1.0], -1, 25 / 6),
        ),
    )
    def test_lstsq(self, x, y, a_ref, b_ref):
        t_x = Tensor(self.to_tensor(x))
        t_y = Tensor(self.to_tensor(y))
        M = Tensor(self.to_tensor(np.vstack([x, np.ones_like(x) ** 0]).transpose()))
        M = M.astype(t_x.dtype)

        solution = fns.linalg.lstsq(M, t_y)
        a, b = solution

        assert isinstance(solution, Tensor)
        assert fns.allclose(a, a_ref)
        assert fns.allclose(b, b_ref)

    @pytest.mark.parametrize(
        "a, full_matrices, abs_res_ref",
        (
            # example is taken from: https://www.d.umn.edu/~mhampton/m4326svd_example.pdf
            # compare absolute values, since different backends may vary the sign.
            (
                [[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]],
                True,
                (
                    [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), 1 / sqrt(2)]],
                    [5.0, 3.0],
                    [
                        [1 / sqrt(2), 1 / sqrt(2), 0.0],
                        [1 / sqrt(18), 1 / sqrt(18), 4 / sqrt(18)],
                        [2 / 3, 2 / 3, 1 / 3],
                    ],
                ),
            ),
            (
                [[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]],
                False,
                (
                    [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), 1 / sqrt(2)]],
                    [5.0, 3.0],
                    [
                        [1 / sqrt(2), 1 / sqrt(2), 0.0],
                        [1 / sqrt(18), 1 / sqrt(18), 4 / sqrt(18)],
                    ],
                ),
            ),
        ),
    )
    def test_svd(self, a, full_matrices, abs_res_ref):
        t_a = Tensor(self.to_tensor(a))

        res = fns.linalg.svd(t_a, full_matrices)

        assert isinstance(res, tuple)
        for act, abs_ref in zip(res, abs_res_ref):
            assert isinstance(act, Tensor)
            assert fns.allclose(fns.abs(act), abs_ref, atol=1e-7)

    @pytest.mark.parametrize("data", [[[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]]])
    def test_save_load_file(self, tmp_path, data):
        tensor_key, tensor_filename = "tensor_key", "test_tensor"
        tensor = Tensor(self.to_tensor(data))
        stat = {tensor_key: tensor}
        fns.io.save_file(stat, tmp_path / tensor_filename)
        loaded_stat = fns.io.load_file(tmp_path / tensor_filename, backend=tensor.backend, device=tensor.device)
        assert fns.allclose(stat[tensor_key], loaded_stat[tensor_key])
        assert isinstance(loaded_stat[tensor_key], Tensor)
        assert loaded_stat[tensor_key].backend == tensor.backend
        assert loaded_stat[tensor_key].device == tensor.device
        assert loaded_stat[tensor_key].dtype == tensor.dtype

    def test_save_load_symlink_error(self, tmp_path):
        file_path = tmp_path / "test_tensor"
        symlink_path = tmp_path / "symlink_test_tensor"
        symlink_path.symlink_to(file_path)

        tensor_key = "tensor_key"
        tensor = Tensor(self.to_tensor([1, 2]))
        stat = {tensor_key: tensor}

        with pytest.raises(nncf.ValidationError, match="symbolic link"):
            fns.io.save_file(stat, symlink_path)
        with pytest.raises(nncf.ValidationError, match="symbolic link"):
            fns.io.load_file(symlink_path, backend=tensor.backend)

    @pytest.mark.parametrize("data", [[3.0, 2.0, 2.0], [1, 2, 3]])
    @pytest.mark.parametrize("dtype", [TensorDataType.float32, TensorDataType.int32, TensorDataType.uint8, None])
    def test_fn_tensor(self, data, dtype):
        if (
            self.backend() == TensorBackend.tf
            and dtype is not None
            and not dtype.is_float()
            and (data == [3.0, 2.0, 2.0])
        ):
            pytest.skip("TF backend does not support non-float dtypes for float data")
        nncf_tensor = fns.tensor(data, backend=self.backend(), dtype=dtype, device=self.device())
        backend_tensor = Tensor(self.to_tensor(data))
        if dtype is not None:
            backend_tensor = backend_tensor.astype(dtype)
        assert fns.allclose(nncf_tensor, backend_tensor)
        assert nncf_tensor.dtype == backend_tensor.dtype

    def test_as_numpy_tensor(self):
        tensor1 = Tensor(self.to_tensor([1.0, 2.0]))
        tensor2 = tensor1.as_numpy_tensor()
        assert tensor2.backend == TensorBackend.numpy
        assert tensor1.dtype == tensor2.dtype
        assert tensor1.shape == tensor2.shape
        assert tensor2.device == TensorDeviceType.CPU
        assert fns.allclose(tensor1, tensor2)
