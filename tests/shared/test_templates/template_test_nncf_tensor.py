# Copyright (c) 2024 Intel Corporation
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
from typing import TypeVar

import pytest

from nncf.experimental.common.tensor_statistics import statistical_functions as s_fns
from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import TensorDataType
from nncf.experimental.tensor import TensorDeviceType
from nncf.experimental.tensor import functions as fns

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


class TemplateTestNNCFTensorOperators:
    @staticmethod
    @abstractmethod
    def to_tensor(x: TTensor) -> TTensor:
        pass

    @staticmethod
    @abstractmethod
    def cast_to(x: TTensor, dtype: TensorDataType) -> TTensor:
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
        assert res_nncf.device == nncf_tensor_a.device

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
        assert res_nncf.device == nncf_tensor_a.device

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

    def test_getitem(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        res = nncf_tensor[1]
        assert res == 1
        assert isinstance(res, Tensor)
        assert res.device == nncf_tensor.device

    def test_iter(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        for i, x in enumerate(nncf_tensor):
            assert x == arr[i]
            assert isinstance(x, Tensor)

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
        "x1, x2, rtol, atol, ref",
        (
            ([0.1], [0.1], None, None, True),
            ([0.1], [0.10001], None, None, False),
            ([0.1], [0.10001], 0.1, None, True),
            ([0.1], [0.10001], None, 0.1, True),
            ([0.1], [0.20001], None, 0.1, False),
            ([0.1], 0.1, None, None, True),
        ),
    )
    def test_fn_allclose(self, x1, x2, rtol, atol, ref):
        tensor1 = Tensor(self.to_tensor(x1))
        tensor2 = Tensor(self.to_tensor(x2))
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
        "val, decimals, ref",
        (
            (1.1, 0, 1.0),
            ([1.1, 0.9], 0, [1.0, 1.0]),
            ([1.11, 0.91], 1, [1.1, 0.9]),
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
        ),
    )
    def test_fn_linalg_norm(self, x, ord, axis, keepdims, ref):
        tensor = Tensor(self.to_tensor(x))
        ref_tensor = self.to_tensor(ref)

        res = fns.linalg.norm(tensor, ord, axis, keepdims)

        assert isinstance(res, Tensor)
        assert fns.allclose(res.data, ref_tensor)
        assert res.device == tensor.device

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
