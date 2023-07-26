# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-many-function-args

import operator
from abc import abstractmethod
from typing import TypeVar

import pytest

from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import TensorDataType
from nncf.experimental.tensor import TensorDeviceType
from nncf.experimental.tensor import functions

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
}

COMPARISON_OPERATOR_MAP = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "ge": operator.ge,
}


# pylint: disable=too-many-public-methods
class TemplateTestNNCFTensorOperators:
    @staticmethod
    @abstractmethod
    def to_tensor(x: TTensor) -> TTensor:
        pass

    @pytest.mark.parametrize("op_name", OPERATOR_MAP.keys())
    def test_operators_tensor(self, op_name):
        tensor_a = self.to_tensor([1, 2])
        tensor_b = self.to_tensor([22, 11])

        nncf_tensor_a = Tensor(tensor_a)
        nncf_tensor_b = Tensor(tensor_b)

        fn = OPERATOR_MAP[op_name]
        res = fn(tensor_a, tensor_b)
        res_nncf = fn(nncf_tensor_a, nncf_tensor_b)

        assert res.dtype == res_nncf.data.dtype
        assert all(res == res_nncf.data)
        assert isinstance(res_nncf, Tensor)

    @pytest.mark.parametrize("op_name", OPERATOR_MAP.keys())
    def test_operators_int(self, op_name):
        tensor_a = self.to_tensor([1, 2])
        value = 2

        nncf_tensor_a = Tensor(tensor_a)

        fn = OPERATOR_MAP[op_name]
        res = fn(tensor_a, value)
        res_nncf = fn(nncf_tensor_a, value)

        assert res.dtype == res_nncf.data.dtype
        assert all(res == res_nncf.data)
        assert isinstance(res_nncf, Tensor)

    @pytest.mark.parametrize("op_name", ("add", "sub", "mul", "truediv", "floordiv"))
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

    @pytest.mark.parametrize("op_name", COMPARISON_OPERATOR_MAP.keys())
    def test_comparison_tensor(self, op_name):
        tensor_a = self.to_tensor((1,))
        tensor_b = self.to_tensor((2,))

        nncf_tensor_a = Tensor(tensor_a)
        nncf_tensor_b = Tensor(tensor_b)

        fn = COMPARISON_OPERATOR_MAP[op_name]
        res = fn(tensor_a, tensor_b)
        res_nncf = fn(nncf_tensor_a, nncf_tensor_b)

        assert res == res_nncf
        assert isinstance(res_nncf, Tensor)

    @pytest.mark.parametrize("op_name", COMPARISON_OPERATOR_MAP.keys())
    def test_comparison_int(self, op_name):
        tensor_a = self.to_tensor((1,))
        value = 2

        nncf_tensor_a = Tensor(tensor_a)

        fn = COMPARISON_OPERATOR_MAP[op_name]
        res = fn(tensor_a, value)
        res_nncf = fn(nncf_tensor_a, value)

        assert res == res_nncf
        assert isinstance(res_nncf, Tensor)

    @pytest.mark.parametrize("op_name", COMPARISON_OPERATOR_MAP.keys())
    def test_comparison_int_rev(self, op_name):
        tensor_a = self.to_tensor((1,))
        value = 2

        nncf_tensor_a = Tensor(tensor_a)

        fn = COMPARISON_OPERATOR_MAP[op_name]
        res = fn(value, tensor_a)
        res_nncf = fn(value, nncf_tensor_a)

        assert res == res_nncf
        assert isinstance(res_nncf, Tensor)

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
    def test_squeeze(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.squeeze(axis=axis)
        if isinstance(ref, list):
            assert functions.all(res == ref_tensor)
        else:
            assert res == ref_tensor
        assert isinstance(res, Tensor)

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
        res = functions.squeeze(nncf_tensor, axis=axis)
        if isinstance(ref, list):
            assert functions.all(res == ref_tensor)
        else:
            assert res == ref_tensor
        assert isinstance(res, Tensor)

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
        if isinstance(ref, list):
            assert all(res.data == ref_tensor)
        else:
            assert res.data == ref_tensor
        assert isinstance(res, Tensor)

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[3], [4]]]], None, 4),
            ([[1, 2], [3, 4]], 1, [2, 4]),
        ),
    )
    def test_max(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.max(axis=axis)
        if isinstance(ref, list):
            assert all(res.data == ref_tensor)
        else:
            assert res.data == ref_tensor
        assert isinstance(res, Tensor)

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[3], [4]]]], None, 4),
            ([[1, 2], [3, 4]], 1, [2, 4]),
        ),
    )
    def test_fn_max(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = functions.max(nncf_tensor, axis=axis)
        if isinstance(ref, list):
            assert all(res.data == ref_tensor)
        else:
            assert res.data == ref_tensor
        assert isinstance(res, Tensor)

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[3], [4]]]], None, 1),
            ([[1, 2], [3, 4]], 1, [1, 3]),
        ),
    )
    def test_min(self, val, axis, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)
        res = nncf_tensor.min(axis=axis)
        if isinstance(ref, list):
            assert all(res.data == ref_tensor)
        else:
            assert res.data == ref_tensor
        assert isinstance(res, Tensor)

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, 1),
            ([1], None, 1),
            ([[[[1], [2]], [[3], [4]]]], None, 1),
            ([[1, 2], [3, 4]], 1, [1, 3]),
        ),
    )
    def test_fn_min(self, val, axis, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        ref_tensor = self.to_tensor(ref)
        res = functions.min(nncf_tensor, axis=axis)
        if isinstance(ref, list):
            assert all(res.data == ref_tensor)
        else:
            assert res.data == ref_tensor
        assert isinstance(res, Tensor)

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
        if isinstance(ref, list):
            assert all(res == nncf_ref_tensor)
        else:
            assert res == nncf_ref_tensor
        assert isinstance(res, Tensor)

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
        res = functions.abs(nncf_tensor)
        if isinstance(ref, list):
            assert all(res == nncf_ref_tensor)
        else:
            assert res == nncf_ref_tensor
        assert isinstance(res, Tensor)

    def test_getitem(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        res = nncf_tensor[1]
        assert res == 1
        assert isinstance(res, Tensor)

    def test_iter(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        i = 0
        for x in nncf_tensor:
            assert x == arr[i]
            assert isinstance(x, Tensor)
            i += 1

    # Math

    @pytest.mark.parametrize(
        "axis, ref",
        (
            (None, 3),
            (0, [2, 1]),
        ),
    )
    def test_fn_count_nonzero(self, axis, ref):
        tensor = self.to_tensor([[1, 2], [1, 0]])
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = functions.count_nonzero(nncf_tensor, axis=axis)
        if axis is None:
            assert res.data == ref_tensor
        else:
            assert all(res.data == self.to_tensor(ref))
        assert isinstance(res, Tensor)

    def test_fn_zeros_like(self):
        tensor = self.to_tensor([1, 2])
        nncf_tensor = Tensor(tensor)

        res = functions.zeros_like(nncf_tensor)
        assert all(res == Tensor(tensor * 0))
        assert isinstance(res, Tensor)

    def test_fn_maximum(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = Tensor(self.to_tensor([2, 1]))
        tensor_ref = self.to_tensor([2, 2])

        res = functions.maximum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)

    def test_fn_maximum_list(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = [2, 1]
        tensor_ref = self.to_tensor([2, 2])

        res = functions.maximum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)

    def test_fn_minimum(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = Tensor(self.to_tensor([2, 1]))
        tensor_ref = self.to_tensor([1, 1])

        res = functions.minimum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)

    def test_fn_minimum_list(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_b = [2, 1]
        tensor_ref = self.to_tensor([1, 1])

        res = functions.minimum(tensor_a, tensor_b)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)

    def test_fn_ones_like(self):
        tensor_a = Tensor(self.to_tensor([1, 2]))
        tensor_ref = self.to_tensor([1, 1])

        res = functions.ones_like(tensor_a)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)

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
        res = functions.all(tensor, axis=axis)
        if isinstance(ref, list):
            assert all(res.data == self.to_tensor(ref))
        else:
            assert res.data == self.to_tensor(ref)
        assert isinstance(res, Tensor)

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
        res = functions.any(tensor, axis=axis)
        if isinstance(ref, list):
            assert all(res.data == self.to_tensor(ref))
        else:
            assert res == ref
        assert isinstance(res, Tensor)

    def test_fn_where(self):
        tensor = Tensor(self.to_tensor([1, -1]))
        tensor_ref = self.to_tensor([1, 0])
        res = functions.where(tensor > 0, 1, 0)
        assert all(res.data == tensor_ref)
        assert isinstance(res, Tensor)

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
        res = functions.isempty(tensor)
        assert res == ref
        assert isinstance(res, Tensor)

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
        assert isinstance(res, Tensor)

    @pytest.mark.parametrize(
        "x1, x2, rtol, atol, ref",
        (
            ([0.1], [0.1], None, None, True),
            ([0.1], [0.10001], None, None, False),
            ([0.1], [0.10001], 0.1, None, True),
            ([0.1], [0.10001], None, 0.1, True),
            ([0.1], [0.20001], None, 0.1, False),
        ),
    )
    def test_fn_allclose(self, x1, x2, rtol, atol, ref):
        tensor1 = Tensor(self.to_tensor(x1))
        tensor2 = Tensor(self.to_tensor(x2))
        if rtol is not None:
            res = functions.allclose(tensor1, tensor2, rtol=rtol)
        elif atol is not None:
            res = functions.allclose(tensor1, tensor2, atol=atol)
        else:
            res = functions.allclose(tensor1, tensor2)
        assert res == ref
        assert isinstance(res, Tensor)

    @pytest.mark.parametrize(
        "x1, x2, rtol, atol, ref",
        (
            ([0.1], [0.1], None, None, [True]),
            ([0.1], [0.10001], None, None, [False]),
            ([0.1], [0.10001], 0.1, None, [True]),
            ([0.1], [0.10001], None, 0.1, [True]),
        ),
    )
    def test_fn_isclose(self, x1, x2, rtol, atol, ref):
        tensor1 = Tensor(self.to_tensor(x1))
        tensor2 = Tensor(self.to_tensor(x2))
        if rtol is not None:
            res = functions.isclose(tensor1, tensor2, rtol=rtol)
        elif atol is not None:
            res = functions.isclose(tensor1, tensor2, atol=atol)
        else:
            res = functions.isclose(tensor1, tensor2)
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

    def test_fn_astype(self):
        tensor = Tensor(self.to_tensor([1]))
        res = functions.astype(tensor, TensorDataType.int8)
        assert isinstance(res, Tensor)
        assert res.dtype == TensorDataType.int8

    def test_reshape(self):
        tensor = Tensor(self.to_tensor([1, 1]))
        assert tensor.shape == [2]
        assert tensor.reshape([1, 2]).shape == [1, 2]

    def test_fn_reshape(self):
        tensor = Tensor(self.to_tensor([1, 1]))
        assert tensor.shape == [2]
        assert functions.reshape(tensor, [1, 2]).shape == [1, 2]

    def test_not_implemented(self):
        with pytest.raises(NotImplementedError, match="is not implemented for"):
            functions.device({}, [1, 2])
