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

import nncf.common.tensor_new.math as nncf_math
from nncf.common.tensor_new.tensor import Tensor

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
            (1, None, []),
            ([1], None, [1]),
            ([[[1], [2]]], None, [1, 2, 1]),
            ([[[1], [2]]], 1, 2),
        ),
    )
    def test_size(self, val, axis, ref):
        nncf_tensor = Tensor(self.to_tensor(val))
        ref_tensor = Tensor(self.to_tensor(ref))
        if isinstance(ref, list):
            assert all(nncf_tensor.size(axis) == ref_tensor)
        else:
            assert nncf_tensor.size(axis) == ref_tensor

    @pytest.mark.parametrize(
        "val, axis, ref",
        (
            (1, None, []),
            ([1], None, []),
            ([[[[1], [2]], [[1], [2]]]], None, [2, 2]),
            ([[[[1], [2]], [[1], [2]]]], 0, [2, 2, 1]),
            ([[[[1], [2]], [[1], [2]]]], -1, [1, 2, 2]),
        ),
    )
    def test_squeeze(self, val, axis, ref):
        tensor = self.to_tensor(val)
        nncf_tensor = Tensor(tensor)
        ref_tensor = Tensor(self.to_tensor(ref))
        if isinstance(ref, list):
            assert all(nncf_tensor.squeeze(axis=axis).size() == ref_tensor)
        else:
            assert nncf_tensor.squeeze(axis=axis).size() == ref_tensor

    @pytest.mark.parametrize(
        "axis, ref",
        (
            (None, 3),
            (0, [2, 1]),
        ),
    )
    def test_count_nonzero(self, axis, ref):
        tensor = self.to_tensor([[1, 2], [1, 0]])
        nncf_tensor = Tensor(tensor)
        nncf_ref_tensor = Tensor(self.to_tensor(ref))

        if axis is None:
            assert nncf_tensor.count_nonzero() == nncf_ref_tensor
        else:
            assert all(nncf_tensor.count_nonzero(axis=axis) == nncf_ref_tensor)

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

    def test_math_zeros_like(self):
        tensor = self.to_tensor([1, 2])
        nncf_tensor = Tensor(tensor)

        res = nncf_math.zeros_like(nncf_tensor)
        assert all(res == Tensor(tensor * 0))

    def test_math_maximum(self):
        tensor_a = self.to_tensor([1, 2])
        tensor_b = self.to_tensor([2, 1])
        tensor_ref = self.to_tensor([2, 2])

        res = nncf_math.maximum(tensor_a, tensor_b)
        assert all(res == tensor_ref)

    def test_math_minimum(self):
        tensor_a = self.to_tensor([1, 2])
        tensor_b = self.to_tensor([2, 1])
        tensor_ref = self.to_tensor([1, 1])

        res = nncf_math.minimum(tensor_a, tensor_b)
        assert all(res == tensor_ref)

    def test_math_ones_like(self):
        tensor_a = self.to_tensor([1, 2])
        tensor_ref = self.to_tensor([1, 1])

        res = nncf_math.ones_like(tensor_a)
        assert all(res == tensor_ref)

    @pytest.mark.parametrize(
        "val, ref",
        (
            ([True, True], True),
            ([True, False], False),
            ([False, False], False),
        ),
    )
    def test_math_all(self, val, ref):
        tensor = self.to_tensor(val)
        res = nncf_math.all(tensor)
        assert res == ref

    @pytest.mark.parametrize(
        "val, ref",
        (
            ([True, True], True),
            ([True, False], True),
            ([False, False], False),
        ),
    )
    def test_math_any(self, val, ref):
        tensor = self.to_tensor(val)
        res = nncf_math.any(tensor)
        assert res == ref

    def test_math_where(self):
        tensor = self.to_tensor([1, -1])
        tensor_ref = self.to_tensor([1, 0])
        res = nncf_math.where(tensor > 0, 1, 0)
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
        res = nncf_math.is_empty(tensor)
        assert res == ref


# def is_empty(target: Tensor) -> Tensor:
#     return tensor_func_dispatcher("is_empty", target)
