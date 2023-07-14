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
from abc import abstractmethod
from typing import TypeVar

import pytest

import nncf.common.tensor_new.functions as nncf_fns
from nncf.common.tensor_new import Tensor

TModel = TypeVar("TModel")
TTensor = TypeVar("TTensor")


class TemplateTestNNCFTensorOperators:
    @staticmethod
    @abstractmethod
    def to_tensor(x: TTensor) -> TTensor:
        pass

    @pytest.mark.parametrize("operator", ("add", "sub", "mul", "pow", "truediv", "floordiv"))
    def test_builtin_operator_tensor(self, operator):
        tensor_a = self.to_tensor([1, 2])
        tensor_b = self.to_tensor([22, 11])

        nncf_tensor_a = Tensor(tensor_a)
        nncf_tensor_b = Tensor(tensor_b)

        if operator == "add":
            res_tensor = tensor_a + tensor_b
            res_nncf_tensor = nncf_tensor_a + nncf_tensor_b
        elif operator == "sub":
            res_tensor = tensor_a - tensor_b
            res_nncf_tensor = nncf_tensor_a - nncf_tensor_b
        elif operator == "mul":
            res_tensor = tensor_a * tensor_b
            res_nncf_tensor = nncf_tensor_a * nncf_tensor_b
        elif operator == "pow":
            res_tensor = tensor_a**tensor_b
            res_nncf_tensor = nncf_tensor_a**nncf_tensor_b
        elif operator == "truediv":
            res_tensor = tensor_a / tensor_b
            res_nncf_tensor = nncf_tensor_a / nncf_tensor_b
        elif operator == "floordiv":
            res_tensor = tensor_a // tensor_b
            res_nncf_tensor = nncf_tensor_a // nncf_tensor_b

        assert res_tensor.dtype == res_nncf_tensor.data.dtype
        assert all(res_tensor == res_nncf_tensor.data)

    @pytest.mark.parametrize(
        "operator", ("add", "radd", "sub", "rsub", "neg", "mul", "rmul", "pow", "truediv", "floordiv")
    )
    def test_builtin_operator_int(self, operator):
        tensor = self.to_tensor([1, 2])

        nncf_tensor = Tensor(tensor)

        if operator == "add":
            res_tensor = tensor + 2
            res_nncf_tensor = nncf_tensor + 2
        if operator == "radd":
            res_tensor = 2 + tensor
            res_nncf_tensor = 2 + nncf_tensor
        elif operator == "sub":
            res_tensor = tensor - 2
            res_nncf_tensor = nncf_tensor - 2
        elif operator == "rsub":
            res_tensor = 2 - tensor
            res_nncf_tensor = 2 - nncf_tensor
        elif operator == "neg":
            res_tensor = -tensor
            res_nncf_tensor = -nncf_tensor
        elif operator == "mul":
            res_tensor = tensor * 2
            res_nncf_tensor = nncf_tensor * 2
        elif operator == "rmul":
            res_tensor = 2 * tensor
            res_nncf_tensor = 2 * nncf_tensor
        elif operator == "pow":
            res_tensor = tensor**2
            res_nncf_tensor = nncf_tensor**2
        elif operator == "truediv":
            res_tensor = tensor / 2
            res_nncf_tensor = nncf_tensor / 2
        elif operator == "floordiv":
            res_tensor = tensor // 2
            res_nncf_tensor = nncf_tensor // 2

        assert res_tensor.dtype == res_nncf_tensor.data.dtype
        assert all(res_tensor == res_nncf_tensor.data)

    @pytest.mark.parametrize("operator", ("lt", "le", "eq", "nq", "gt", "ge"))
    def test_comparison_operator_tensor(self, operator):
        tensor_a = self.to_tensor((1,))
        tensor_b = self.to_tensor((2,))

        nncf_tensor_a = Tensor(tensor_a)
        nncf_tensor_b = Tensor(tensor_b)

        if operator == "lt":
            res = tensor_a < tensor_b
            res_nncf = nncf_tensor_a < nncf_tensor_b
        if operator == "le":
            res = tensor_a <= tensor_b
            res_nncf = nncf_tensor_a <= nncf_tensor_b
        if operator == "eq":
            res = tensor_a == tensor_b
            res_nncf = nncf_tensor_a == nncf_tensor_b
        if operator == "nq":
            res = tensor_a != tensor_b
            res_nncf = nncf_tensor_a != nncf_tensor_b
        if operator == "gt":
            res = tensor_a > tensor_b
            res_nncf = nncf_tensor_a > nncf_tensor_b
        if operator == "ge":
            res = tensor_a >= tensor_b
            res_nncf = nncf_tensor_a >= nncf_tensor_b

        assert res == res_nncf

    @pytest.mark.parametrize("operator", ("lt", "le", "gt", "ge"))
    def test_comparison_operator_int(self, operator):
        tensor = self.to_tensor((1,))
        nncf_tensor = Tensor(tensor)

        if operator == "lt":
            res = tensor < 2
            res_nncf = nncf_tensor < 2
        if operator == "le":
            res = tensor <= 2
            res_nncf = nncf_tensor <= 2
        if operator == "gt":
            res = tensor > 2
            res_nncf = nncf_tensor > 2
        if operator == "ge":
            res = tensor >= 2
            res_nncf = nncf_tensor >= 2

        assert res == res_nncf

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
            assert nncf_fns.all(res == ref_tensor)
        else:
            assert res == ref_tensor

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
            assert nncf_fns.all(res == ref_tensor)
        else:
            assert res == ref_tensor

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
        ref_tensor = Tensor(self.to_tensor(ref))
        if isinstance(ref, list):
            assert all(nncf_tensor.max(axis=axis) == ref_tensor)
        else:
            assert nncf_tensor.max(axis=axis) == ref_tensor

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
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = Tensor(self.to_tensor(ref))
        if isinstance(ref, list):
            assert all(nncf_tensor.min(axis=axis) == ref_tensor)
        else:
            assert nncf_tensor.min(axis=axis) == ref_tensor

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

        if isinstance(ref, list):
            assert all(nncf_tensor.abs() == nncf_ref_tensor)
        else:
            assert nncf_tensor.abs() == nncf_ref_tensor

    def test_getitem(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))

        assert nncf_tensor[1] == 1

    def test_iter(self):
        arr = [0, 1, 2]
        nncf_tensor = Tensor(self.to_tensor(arr))
        i = 0
        for x in nncf_tensor:
            assert x == arr[i]
            i += 1

    # Math

    @pytest.mark.parametrize(
        "axis, ref",
        (
            (None, 3),
            (0, [2, 1]),
        ),
    )
    def test_math_count_nonzero(self, axis, ref):
        tensor = self.to_tensor([[1, 2], [1, 0]])
        nncf_tensor = Tensor(tensor)
        nncf_ref_tensor = Tensor(self.to_tensor(ref))

        if axis is None:
            assert nncf_fns.count_nonzero(nncf_tensor) == nncf_ref_tensor
        else:
            assert all(nncf_fns.count_nonzero(nncf_tensor, axis=axis) == nncf_ref_tensor)

    def test_math_zeros_like(self):
        tensor = self.to_tensor([1, 2])
        nncf_tensor = Tensor(tensor)

        res = nncf_fns.zeros_like(nncf_tensor)
        assert all(res == Tensor(tensor * 0))

    def test_math_maximum(self):
        tensor_a = self.to_tensor([1, 2])
        tensor_b = self.to_tensor([2, 1])
        tensor_ref = self.to_tensor([2, 2])

        res = nncf_fns.maximum(tensor_a, tensor_b)
        assert all(res == tensor_ref)

    def test_math_minimum(self):
        tensor_a = self.to_tensor([1, 2])
        tensor_b = self.to_tensor([2, 1])
        tensor_ref = self.to_tensor([1, 1])

        res = nncf_fns.minimum(tensor_a, tensor_b)
        assert all(res == tensor_ref)

    def test_math_ones_like(self):
        tensor_a = self.to_tensor([1, 2])
        tensor_ref = self.to_tensor([1, 1])

        res = nncf_fns.ones_like(tensor_a)
        assert all(res == tensor_ref)

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            ([True, True], None, True),
            ([True, False], None, False),
            ([False, False], None, False),
            ([[True, True], [False, True]], 0, [False, True]),
        ),
    )
    def test_math_all(self, val, axis, ref):
        tensor = self.to_tensor(val)
        res = nncf_fns.all(tensor, axis=axis)
        if isinstance(ref, list):
            assert all(res.data == self.to_tensor(ref))
        else:
            assert res == ref

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            ([True, True], None, True),
            ([True, False], None, True),
            ([False, False], None, False),
            ([[False, True], [False, False]], 0, [False, True]),
        ),
    )
    def test_math_any(self, val, axis, ref):
        tensor = self.to_tensor(val)
        res = nncf_fns.any(tensor, axis=axis)
        if isinstance(ref, list):
            assert all(res.data == self.to_tensor(ref))
        else:
            assert res == ref

    def test_math_where(self):
        tensor = self.to_tensor([1, -1])
        tensor_ref = self.to_tensor([1, 0])
        res = nncf_fns.where(tensor > 0, 1, 0)
        assert all(res == tensor_ref)

    @pytest.mark.parametrize(
        "val, ref",
        (
            ([], True),
            ([1], False),
            (1, False),
        ),
    )
    def test_math_is_empty(self, val, ref):
        tensor = self.to_tensor(val)
        res = nncf_fns.is_empty(tensor)
        assert res == ref

    @pytest.mark.parametrize(
        "x1, x2, rtol, atol, ref",
        (
            ([0.1], [0.1], None, None, True),
            ([0.1], [0.10001], None, None, False),
            ([0.1], [0.10001], 0.1, None, True),
            ([0.1], [0.10001], None, 0.1, True),
        ),
    )
    def test_math_allclose(self, x1, x2, rtol, atol, ref):
        if rtol is not None:
            res = nncf_fns.allclose(self.to_tensor(x1), self.to_tensor(x2), rtol=rtol)
        elif atol is not None:
            res = nncf_fns.allclose(self.to_tensor(x1), self.to_tensor(x2), atol=atol)
        else:
            res = nncf_fns.allclose(self.to_tensor(x1), self.to_tensor(x2))
        assert res == ref

    @pytest.mark.parametrize(
        "x1, x2, rtol, atol, ref",
        (
            ([0.1], [0.1], None, None, [True]),
            ([0.1], [0.10001], None, None, [False]),
            ([0.1], [0.10001], 0.1, None, [True]),
            ([0.1], [0.10001], None, 0.1, [True]),
        ),
    )
    def test_math_isclose(self, x1, x2, rtol, atol, ref):
        if rtol is not None:
            res = nncf_fns.isclose(self.to_tensor(x1), self.to_tensor(x2), rtol=rtol)
        elif atol is not None:
            res = nncf_fns.isclose(self.to_tensor(x1), self.to_tensor(x2), atol=atol)
        else:
            res = nncf_fns.isclose(self.to_tensor(x1), self.to_tensor(x2))
        assert all(res == self.to_tensor(ref))
