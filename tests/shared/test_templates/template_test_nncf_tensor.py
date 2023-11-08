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
        tensor = self.to_tensor([[1.0, 2.0], [1.0, 0.0]])
        nncf_tensor = Tensor(tensor)
        ref_tensor = self.to_tensor(ref)
        res = fns.count_nonzero(nncf_tensor, axis=axis)

        assert isinstance(res, Tensor)
        assert fns.allclose(res, ref_tensor)
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
        assert fns.allclose(res, self.to_tensor(ref))
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
        assert fns.allclose(res, self.to_tensor(ref))
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
            fns.device(Tensor(None))

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
        assert fns.all(res == ref)
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
        assert fns.allclose(res, ref_tensor)
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
        assert fns.allclose(res, ref_tensor)
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
