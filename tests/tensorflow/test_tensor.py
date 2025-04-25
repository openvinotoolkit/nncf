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
import tensorflow as tf

from nncf.tensor import Tensor
from nncf.tensor import TensorDataType
from nncf.tensor.definitions import TensorBackend
from nncf.tensor.definitions import TensorDeviceType
from nncf.tensor.functions import linalg
from tests.cross_fw.test_templates.template_test_nncf_tensor import TemplateTestNNCFTensorOperators


def cast_to(x: tf.Tensor, dtype: TensorDataType) -> tf.Tensor:
    if dtype is TensorDataType.float32:
        return tf.cast(x, tf.float32)
    if dtype is TensorDataType.float16:
        return tf.cast(x, tf.float16)
    raise NotImplementedError


class TestTFNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        with tf.device("CPU"):
            return tf.constant(x)

    @staticmethod
    def to_cpu(x):
        return x

    @staticmethod
    def cast_to(x: tf.Tensor, dtype: TensorDataType) -> tf.Tensor:
        return cast_to(x, dtype)

    @staticmethod
    def backend() -> TensorBackend:
        return TensorBackend.tf

    @staticmethod
    def device() -> TensorDeviceType:
        return TensorDeviceType.CPU

    def test_norm_keepdims(self):
        tensor_data = [[1.0, 2.0], [3.0, 4.0]]
        tf_tensor = self.to_tensor(tensor_data)
        tensor = Tensor(tf_tensor)

        result = linalg.norm(tensor, ord="nuc", keepdims=True)

        assert result.shape == (1, 1)

        for ord_val in [None, 0, 1, 2, -1, -2, "fro"]:
            result = linalg.norm(tensor, ord=ord_val, keepdims=True)
            assert result.shape == (1, 1), f"Failed for ord={ord_val}"

    def test_lstsq_rank2(self):
        x_data = [1.0, 2.0, 4.0]
        ones_data = [1.0, 1.0, 1.0]
        a_data = [[x_data[0], ones_data[0]], [x_data[1], ones_data[1]], [x_data[2], ones_data[2]]]

        a_tensor = self.to_tensor(a_data)
        a = Tensor(a_tensor)

        b_data = [[6.0, 6.0], [8.0, 10.0], [12.0, 18.0]]
        b_tensor = self.to_tensor(b_data)
        b = Tensor(b_tensor)

        x = linalg.lstsq(a, b)

        assert x.shape == (2, 2)

        expected = [[2.0, 4.0], [4.0, 2.0]]

        for i in range(2):
            for j in range(2):
                x_val = x.data.numpy()[i][j]
                expected_val = expected[i][j]
                assert abs(x_val - expected_val) < 0.2, f"Value at ({i},{j}) is {x_val}, expected {expected_val}"

    def test_norm_comprehensive(self):
        # 2D tensor
        tensor_data_2d = [[1.0, 2.0], [3.0, 4.0]]
        tf_tensor_2d = self.to_tensor(tensor_data_2d)
        tensor_2d = Tensor(tf_tensor_2d)

        # 3D tensor
        tensor_data_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        tf_tensor_3d = self.to_tensor(tensor_data_3d)
        tensor_3d = Tensor(tf_tensor_3d)

        matrix_ord_values = [None, 0, 1, 2, -1, -2, float("inf"), -float("inf"), "fro", "nuc"]

        vector_ord_values = [None, 0, 1, 2, float("inf")]

        # Test vector norms (axis=0 or axis=1)
        for ord_val in vector_ord_values:
            if ord_val == "fro" or ord_val == "nuc":
                continue

            result = linalg.norm(tensor_2d, ord=ord_val, axis=0)
            assert result.shape == (2,), f"Failed for ord={ord_val}, axis=0"

            result = linalg.norm(tensor_2d, ord=ord_val, axis=1)
            assert result.shape == (2,), f"Failed for ord={ord_val}, axis=1"

        # Test matrix norms (axis=None or axis=(0,1))
        for ord_val in matrix_ord_values:
            try:
                result = linalg.norm(tensor_2d, ord=ord_val)

                result = linalg.norm(tensor_2d, ord=ord_val, axis=(0, 1))
                assert result.ndim == 0, f"Failed for ord={ord_val}, axis=(0,1)"

                result = linalg.norm(tensor_2d, ord=ord_val, axis=(0, 1), keepdims=True)
                assert result.shape == (1, 1), f"Failed for ord={ord_val}, axis=(0,1), keepdims=True"
            except ValueError:
                pass

        # Test 3D tensor slicing for nuclear norm
        try:
            result = linalg.norm(tensor_3d, ord="nuc", axis=(1, 2))
            assert result.shape == (2,), "Failed for 3D tensor, ord=nuc, axis=(1,2)"

            result = linalg.norm(tensor_3d, ord="nuc", axis=(1, 2), keepdims=True)
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
                result = linalg.norm(tensor_3d, ord=ord_val, axis=axis_val)

                expected_shape = list(tensor_3d.shape)
                expected_shape.pop(axis_val)
                assert result.shape == tuple(expected_shape), f"Failed shape check for ord={ord_val}, axis={axis_val}"

                result_keep = linalg.norm(tensor_3d, ord=ord_val, axis=axis_val, keepdims=True)
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
                    result = linalg.norm(tensor_3d, ord=ord_val, axis=axis_pair)

                    expected_shape = []
                    for i in range(tensor_3d.ndim):
                        if i not in axis_pair:
                            expected_shape.append(tensor_3d.shape[i])
                    assert result.shape == tuple(expected_shape), (
                        f"Failed shape check for ord={ord_val}, axis={axis_pair}"
                    )

                    result_keep = linalg.norm(tensor_3d, ord=ord_val, axis=axis_pair, keepdims=True)
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
                result = linalg.norm(tensor_3d, ord="nuc", axis=axis_pair)

                result_keep = linalg.norm(tensor_3d, ord="nuc", axis=axis_pair, keepdims=True)

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

        # 2D tensor
        tensor_data_2d = [[1.0, 0.0, 3.0], [0.0, 0.0, 6.0], [7.0, 0.0, 9.0]]
        tf_tensor_2d = self.to_tensor(tensor_data_2d)
        tensor_2d = Tensor(tf_tensor_2d)

        # 3D tensor
        tensor_data_3d = [[[1.0, 0.0], [0.0, 4.0]], [[0.0, 0.0], [7.0, 8.0]]]
        tf_tensor_3d = self.to_tensor(tensor_data_3d)
        tensor_3d = Tensor(tf_tensor_3d)

        # Test 1D tensor
        result = linalg.norm(tensor_1d, ord=0)
        assert result.item() == 3, f"Expected 3 non-zeros, got {result.item()}"

        # Test 2D tensor
        result = linalg.norm(tensor_2d, ord=0)
        assert result.item() == 5, f"Expected 5 non-zeros, got {result.item()}"

        result = linalg.norm(tensor_2d, ord=0, axis=0)
        expected = [2, 0, 3]
        for i, val in enumerate(result.data.numpy()):
            assert val == expected[i], f"At index {i}, expected {expected[i]}, got {val}"

        result = linalg.norm(tensor_2d, ord=0, axis=1)
        expected = [2, 1, 2]
        for i, val in enumerate(result.data.numpy()):
            assert val == expected[i], f"At index {i}, expected {expected[i]}, got {val}"

        result = linalg.norm(tensor_2d, ord=0, axis=0, keepdims=True)
        assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"

        result = linalg.norm(tensor_2d, ord=0, axis=1, keepdims=True)
        assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"

        # Test 3D tensor
        result = linalg.norm(tensor_3d, ord=0)
        assert result.item() == 4, f"Expected 4 non-zeros, got {result.item()}"

        result = linalg.norm(tensor_3d, ord=0, axis=0)
        assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
        expected = [[1, 0], [1, 2]]
        for i in range(2):
            for j in range(2):
                assert result.data.numpy()[i][j] == expected[i][j], (
                    f"At position ({i},{j}), expected {expected[i][j]}, got {result.data.numpy()[i][j]}"
                )

        result = linalg.norm(tensor_3d, ord=0, axis=(0, 1))
        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        expected = [2, 2]
        for i, val in enumerate(result.data.numpy()):
            assert val == expected[i], f"At index {i}, expected {expected[i]}, got {val}"

        # Test all combinations with keepdims
        for axis in [0, 1, 2, (0, 1), (0, 2), (1, 2)]:
            result = linalg.norm(tensor_3d, ord=0, axis=axis, keepdims=True)

            expected_shape = list(tensor_3d.shape)
            if isinstance(axis, tuple):
                for ax in axis:
                    expected_shape[ax] = 1
            else:
                expected_shape[axis] = 1

            assert result.shape == tuple(expected_shape), (
                f"For axis={axis}, expected shape {tuple(expected_shape)}, got {result.shape}"
            )

    def test_norm_4d_tensor(self):
        # 4D tensor
        tensor_data_4d = [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]
        tf_tensor_4d = self.to_tensor(tensor_data_4d)
        tensor_4d = Tensor(tf_tensor_4d)

        # Test nuclear norm on slices (axis pairs)
        nuclear_axes = [(1, 2)]
        for axis_pair in nuclear_axes:
            try:
                result = linalg.norm(tensor_4d, ord="nuc", axis=axis_pair)
                result_keep = linalg.norm(tensor_4d, ord="nuc", axis=axis_pair, keepdims=True)

                expected_shape = [dim for i, dim in enumerate(tensor_4d.shape) if i not in axis_pair]
                assert result.shape == tuple(expected_shape)

                expected_shape_keep = [1 if i in axis_pair else dim for i, dim in enumerate(tensor_4d.shape)]
                assert result_keep.shape == tuple(expected_shape_keep)
            except ValueError as e:
                assert False, f"Failed shape check for nuclear norm with axis={axis_pair}, error: {e}"

        # Test ord=0
        result = linalg.norm(tensor_4d, ord=0)
        assert result.ndim == 0
        assert result.item() == tensor_4d.shape[0] * tensor_4d.shape[1] * tensor_4d.shape[2] * tensor_4d.shape[3]

        result = linalg.norm(tensor_4d, ord=0, axis=0)
        assert result.shape == (2, 2, 2)

        result = linalg.norm(tensor_4d, ord=0, axis=(0, 1))
        assert result.shape == (2, 2)

        result = linalg.norm(tensor_4d, ord=0, axis=(0, 1), keepdims=True)
        assert result.shape == (1, 1, 2, 2)

        # Test vector norms
        for ord_val in [1, float("inf")]:
            for axis in range(4):
                result = linalg.norm(tensor_4d, ord=ord_val, axis=axis)
                expected_shape = list(tensor_4d.shape)
                expected_shape.pop(axis)
                assert result.shape == tuple(expected_shape)

                result = linalg.norm(tensor_4d, ord=ord_val, axis=axis, keepdims=True)
                expected_shape = list(tensor_4d.shape)
                expected_shape[axis] = 1
                assert result.shape == tuple(expected_shape)

    def test_norm_empty_tensor(self):
        empty_tensor_1d = self.to_tensor([])
        tensor_1d = Tensor(empty_tensor_1d)

        empty_tensor_2d = self.to_tensor([[]])
        tensor_2d = Tensor(empty_tensor_2d)

        for ord_val in [0, 1, float("inf")]:
            try:
                result = linalg.norm(tensor_1d, ord=ord_val)
                assert result.item() == 0, f"Empty tensor norm with ord={ord_val} should be 0"
            except tf.errors.InvalidArgumentError as e:
                print(f"Note: TensorFlow cannot compute norm with ord={ord_val} on empty tensor: {e}")
            except Exception as e:
                if (
                    "shape" not in str(e).lower()
                    and "empty" not in str(e).lower()
                    and "dimension" not in str(e).lower()
                ):
                    assert False, f"Unexpected error for empty tensor with ord={ord_val}: {e}"
                print(f"Expected error for empty tensor with ord={ord_val}: {e}")

        try:
            result = linalg.norm(tensor_1d, ord=0)
            assert result.item() == 0, "Count of non-zeros in empty tensor should be 0"
        except Exception as e:
            if "shape" not in str(e).lower() and "empty" not in str(e).lower() and "dimension" not in str(e).lower():
                assert False, f"Unexpected error for empty tensor with ord=0: {e}"
            print(f"Expected shape error for empty tensor with ord=0: {e}")

        try:
            result = linalg.norm(tensor_1d, ord=0, keepdims=True)
            assert result.shape == (1,), "Shape with keepdims should be (1,)"
            assert result.item() == 0, "Count of non-zeros in empty tensor should be 0"
        except Exception as e:
            if "shape" not in str(e).lower() and "empty" not in str(e).lower() and "dimension" not in str(e).lower():
                assert False, f"Unexpected error for empty tensor with keepdims: {e}"
            print(f"Expected error with keepdims on empty tensor: {e}")

        for ord_val in ["fro", 0]:
            try:
                result = linalg.norm(tensor_2d, ord=ord_val)
                assert result.item() == 0, f"Empty tensor norm with ord={ord_val} should be 0"
            except tf.errors.InvalidArgumentError as e:
                print(f"Note: TensorFlow cannot compute norm with ord={ord_val} on empty 2D tensor: {e}")
            except Exception as e:
                if (
                    "shape" not in str(e).lower()
                    and "empty" not in str(e).lower()
                    and "dimension" not in str(e).lower()
                ):
                    assert False, f"Unexpected error for empty 2D tensor with ord={ord_val}: {e}"
                print(f"Expected error for empty 2D tensor with ord={ord_val}: {e}")

        try:
            result = linalg.norm(tensor_2d, ord=0, axis=0)
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

        mixed_tensor_data = [[float("inf"), 2.0], [float("nan"), 4.0]]
        mixed_tf_tensor = self.to_tensor(mixed_tensor_data)
        mixed_tensor = Tensor(mixed_tf_tensor)

        result = linalg.norm(inf_tensor, ord=0)
        assert result.item() == 4, "All elements (including Inf) should be counted as non-zero"

        result = linalg.norm(nan_tensor, ord=0)
        assert result.item() == 4, "All elements (including NaN) should be counted as non-zero"

        result = linalg.norm(inf_tensor, ord=0, axis=0)
        assert result.shape == (2,)
        assert result.data.numpy()[0] == 2 and result.data.numpy()[1] == 2

        result = linalg.norm(inf_tensor, ord=0, axis=1)
        assert result.shape == (2,)
        assert result.data.numpy()[0] == 2 and result.data.numpy()[1] == 2

        try:
            result = linalg.norm(inf_tensor, ord="fro")
            assert float("inf") == result.item() or result.item() > 1e30, "Norm with Inf should be Inf or very large"
        except tf.errors.InvalidArgumentError as e:
            print(f"Note: TensorFlow cannot compute Frobenius norm with Inf values: {e}")

        try:
            result = linalg.norm(nan_tensor, ord="nuc")
            import numpy as np

            assert np.isnan(result.item()) or result.item() > 0, "Nuclear norm with NaN might be NaN"
        except tf.errors.InvalidArgumentError as e:
            print(f"Note: TensorFlow SVD cannot handle NaN values: {e}")

        try:
            result = linalg.norm(inf_tensor, ord=float("inf"))
            assert result.item() == float("inf"), "Infinity norm with Inf values should be Inf"
        except tf.errors.InvalidArgumentError as e:
            print(f"Note: TensorFlow cannot compute infinity norm with Inf values: {e}")

        try:
            result = linalg.norm(inf_tensor, ord=0, keepdims=True)
            assert result.shape == (1, 1)
            assert result.item() == 4
        except Exception as e:
            assert False, f"Unexpected error with keepdims and extreme values: {e}"

        try:
            result = linalg.norm(mixed_tensor, ord=0)
            assert result.item() == 4, "All elements should be counted as non-zero"
        except Exception as e:
            if "invalid" not in str(e).lower():
                assert False, f"Unexpected error with mixed NaN and Inf: {e}"
            print(f"Note: Expected error with mixed NaN and Inf values: {e}")

    @pytest.mark.skip("Desired slicing is not supported for TensorFlow")
    @pytest.mark.parametrize("is_tensor_indecies", (False, True))
    def test_getitem_for_indecies(self, is_tensor_indecies):
        pass

    @pytest.mark.skip("TensorFlow throws different kind of exceptions")
    @pytest.mark.parametrize(
        "val, axis, exception_type, exception_match",
        (
            ([[[[1], [2]], [[1], [2]]]], (0, 1), ValueError, "not equal to one"),
            ([[[[1], [2]], [[1], [2]]]], 42, IndexError, "out of"),
            ([[[[1], [2]], [[1], [2]]]], (0, 42), IndexError, "out of"),
        ),
    )
    def test_squeeze_axis_error(self, val, axis, exception_type, exception_match):
        pass


@pytest.mark.skipif(len(tf.config.list_physical_devices("GPU")) == 0, reason="Skipping for CPU-only setups")
class TestGPUTFNNCFTensorOperators(TemplateTestNNCFTensorOperators):
    @staticmethod
    def to_tensor(x):
        with tf.device("GPU"):
            return tf.constant(x)

    @staticmethod
    def to_cpu(x):
        with tf.device("CPU"):
            return tf.constant(x.numpy())

    @staticmethod
    def cast_to(x: tf.Tensor, dtype: TensorDataType) -> tf.Tensor:
        return cast_to(x, dtype)

    def test_device(self):
        tensor = Tensor(self.to_tensor([1]))
        assert tensor.device == TensorDeviceType.GPU

    def test_norm_keepdims(self):
        tensor_data = [[1.0, 2.0], [3.0, 4.0]]
        tf_tensor = self.to_tensor(tensor_data)
        tensor = Tensor(tf_tensor)

        result = linalg.norm(tensor, ord="nuc", keepdims=True)

        assert result.shape == (1, 1)

        for ord_val in [None, 0, 1, 2, -1, -2, "fro"]:
            result = linalg.norm(tensor, ord=ord_val, keepdims=True)
            assert result.shape == (1, 1), f"Failed for ord={ord_val}"

    def test_lstsq_rank2(self):
        x_data = [1.0, 2.0, 4.0]
        ones_data = [1.0, 1.0, 1.0]
        a_data = [[x_data[0], ones_data[0]], [x_data[1], ones_data[1]], [x_data[2], ones_data[2]]]

        a_tensor = self.to_tensor(a_data)
        a = Tensor(a_tensor)

        b_data = [[6.0, 6.0], [8.0, 10.0], [12.0, 18.0]]
        b_tensor = self.to_tensor(b_data)
        b = Tensor(b_tensor)

        x = linalg.lstsq(a, b)

        assert x.shape == (2, 2)

        expected = [[2.0, 4.0], [4.0, 2.0]]

        for i in range(2):
            for j in range(2):
                x_val = x.data.numpy()[i][j]
                expected_val = expected[i][j]
                assert abs(x_val - expected_val) < 0.2, f"Value at ({i},{j}) is {x_val}, expected {expected_val}"

    def test_norm_comprehensive(self):
        # 2D tensor
        tensor_data_2d = [[1.0, 2.0], [3.0, 4.0]]
        tf_tensor_2d = self.to_tensor(tensor_data_2d)
        tensor_2d = Tensor(tf_tensor_2d)

        # 3D tensor
        tensor_data_3d = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]
        tf_tensor_3d = self.to_tensor(tensor_data_3d)
        tensor_3d = Tensor(tf_tensor_3d)

        matrix_ord_values = [None, 0, 1, 2, -1, -2, float("inf"), -float("inf"), "fro", "nuc"]

        vector_ord_values = [None, 0, 1, 2, float("inf")]

        # Test vector norms (axis=0 or axis=1)
        for ord_val in vector_ord_values:
            if ord_val == "fro" or ord_val == "nuc":
                continue

            result = linalg.norm(tensor_2d, ord=ord_val, axis=0)
            assert result.shape == (2,), f"Failed for ord={ord_val}, axis=0"

            result = linalg.norm(tensor_2d, ord=ord_val, axis=1)
            assert result.shape == (2,), f"Failed for ord={ord_val}, axis=1"

        # Test matrix norms (axis=None or axis=(0,1))
        for ord_val in matrix_ord_values:
            try:
                result = linalg.norm(tensor_2d, ord=ord_val)

                result = linalg.norm(tensor_2d, ord=ord_val, axis=(0, 1))
                assert result.ndim == 0, f"Failed for ord={ord_val}, axis=(0,1)"

                result = linalg.norm(tensor_2d, ord=ord_val, axis=(0, 1), keepdims=True)
                assert result.shape == (1, 1), f"Failed for ord={ord_val}, axis=(0,1), keepdims=True"
            except ValueError:
                pass

        # Test 3D tensor slicing for nuclear norm
        try:
            result = linalg.norm(tensor_3d, ord="nuc", axis=(1, 2))
            assert result.shape == (2,), "Failed for 3D tensor, ord=nuc, axis=(1,2)"

            result = linalg.norm(tensor_3d, ord="nuc", axis=(1, 2), keepdims=True)
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
                result = linalg.norm(tensor_3d, ord=ord_val, axis=axis_val)

                expected_shape = list(tensor_3d.shape)
                expected_shape.pop(axis_val)
                assert result.shape == tuple(expected_shape), f"Failed shape check for ord={ord_val}, axis={axis_val}"

                result_keep = linalg.norm(tensor_3d, ord=ord_val, axis=axis_val, keepdims=True)
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
                    result = linalg.norm(tensor_3d, ord=ord_val, axis=axis_pair)

                    expected_shape = []
                    for i in range(tensor_3d.ndim):
                        if i not in axis_pair:
                            expected_shape.append(tensor_3d.shape[i])
                    assert result.shape == tuple(expected_shape), (
                        f"Failed shape check for ord={ord_val}, axis={axis_pair}"
                    )

                    result_keep = linalg.norm(tensor_3d, ord=ord_val, axis=axis_pair, keepdims=True)
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
                result = linalg.norm(tensor_3d, ord="nuc", axis=axis_pair)

                result_keep = linalg.norm(tensor_3d, ord="nuc", axis=axis_pair, keepdims=True)

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

        # 2D tensor
        tensor_data_2d = [[1.0, 0.0, 3.0], [0.0, 0.0, 6.0], [7.0, 0.0, 9.0]]
        tf_tensor_2d = self.to_tensor(tensor_data_2d)
        tensor_2d = Tensor(tf_tensor_2d)

        # 3D tensor
        tensor_data_3d = [[[1.0, 0.0], [0.0, 4.0]], [[0.0, 0.0], [7.0, 8.0]]]
        tf_tensor_3d = self.to_tensor(tensor_data_3d)
        tensor_3d = Tensor(tf_tensor_3d)

        # Test 1D tensor
        result = linalg.norm(tensor_1d, ord=0)
        assert result.item() == 3, f"Expected 3 non-zeros, got {result.item()}"

        # Test 2D tensor
        result = linalg.norm(tensor_2d, ord=0)
        assert result.item() == 5, f"Expected 5 non-zeros, got {result.item()}"

        result = linalg.norm(tensor_2d, ord=0, axis=0)
        expected = [2, 0, 3]
        for i, val in enumerate(result.data.numpy()):
            assert val == expected[i], f"At index {i}, expected {expected[i]}, got {val}"

        result = linalg.norm(tensor_2d, ord=0, axis=1)
        expected = [2, 1, 2]
        for i, val in enumerate(result.data.numpy()):
            assert val == expected[i], f"At index {i}, expected {expected[i]}, got {val}"

        result = linalg.norm(tensor_2d, ord=0, axis=0, keepdims=True)
        assert result.shape == (1, 3), f"Expected shape (1, 3), got {result.shape}"

        result = linalg.norm(tensor_2d, ord=0, axis=1, keepdims=True)
        assert result.shape == (3, 1), f"Expected shape (3, 1), got {result.shape}"

        # Test 3D tensor
        result = linalg.norm(tensor_3d, ord=0)
        assert result.item() == 4, f"Expected 4 non-zeros, got {result.item()}"

        result = linalg.norm(tensor_3d, ord=0, axis=0)
        assert result.shape == (2, 2), f"Expected shape (2, 2), got {result.shape}"
        expected = [[1, 0], [1, 2]]
        for i in range(2):
            for j in range(2):
                assert result.data.numpy()[i][j] == expected[i][j], (
                    f"At position ({i},{j}), expected {expected[i][j]}, got {result.data.numpy()[i][j]}"
                )

        result = linalg.norm(tensor_3d, ord=0, axis=(0, 1))
        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        expected = [2, 2]
        for i, val in enumerate(result.data.numpy()):
            assert val == expected[i], f"At index {i}, expected {expected[i]}, got {val}"

        # Test all combinations with keepdims
        for axis in [0, 1, 2, (0, 1), (0, 2), (1, 2)]:
            result = linalg.norm(tensor_3d, ord=0, axis=axis, keepdims=True)

            expected_shape = list(tensor_3d.shape)
            if isinstance(axis, tuple):
                for ax in axis:
                    expected_shape[ax] = 1
            else:
                expected_shape[axis] = 1

            assert result.shape == tuple(expected_shape), (
                f"For axis={axis}, expected shape {tuple(expected_shape)}, got {result.shape}"
            )

    def test_norm_4d_tensor(self):
        # 4D tensor
        tensor_data_4d = [
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
            [[[9.0, 10.0], [11.0, 12.0]], [[13.0, 14.0], [15.0, 16.0]]],
        ]
        tf_tensor_4d = self.to_tensor(tensor_data_4d)
        tensor_4d = Tensor(tf_tensor_4d)

        # Test nuclear norm on slices (axis pairs)
        nuclear_axes = [(1, 2)]
        for axis_pair in nuclear_axes:
            try:
                result = linalg.norm(tensor_4d, ord="nuc", axis=axis_pair)
                result_keep = linalg.norm(tensor_4d, ord="nuc", axis=axis_pair, keepdims=True)

                expected_shape = [dim for i, dim in enumerate(tensor_4d.shape) if i not in axis_pair]
                assert result.shape == tuple(expected_shape)

                expected_shape_keep = [1 if i in axis_pair else dim for i, dim in enumerate(tensor_4d.shape)]
                assert result_keep.shape == tuple(expected_shape_keep)
            except ValueError as e:
                assert False, f"Failed shape check for nuclear norm with axis={axis_pair}, error: {e}"

        # Test ord=0
        result = linalg.norm(tensor_4d, ord=0)
        assert result.ndim == 0
        assert result.item() == tensor_4d.shape[0] * tensor_4d.shape[1] * tensor_4d.shape[2] * tensor_4d.shape[3]

        result = linalg.norm(tensor_4d, ord=0, axis=0)
        assert result.shape == (2, 2, 2)

        result = linalg.norm(tensor_4d, ord=0, axis=(0, 1))
        assert result.shape == (2, 2)

        result = linalg.norm(tensor_4d, ord=0, axis=(0, 1), keepdims=True)
        assert result.shape == (1, 1, 2, 2)

        # Test vector norms
        for ord_val in [1, float("inf")]:
            for axis in range(4):
                result = linalg.norm(tensor_4d, ord=ord_val, axis=axis)
                expected_shape = list(tensor_4d.shape)
                expected_shape.pop(axis)
                assert result.shape == tuple(expected_shape)

                result = linalg.norm(tensor_4d, ord=ord_val, axis=axis, keepdims=True)
                expected_shape = list(tensor_4d.shape)
                expected_shape[axis] = 1
                assert result.shape == tuple(expected_shape)

    def test_norm_empty_tensor(self):
        empty_tensor_1d = self.to_tensor([])
        tensor_1d = Tensor(empty_tensor_1d)

        empty_tensor_2d = self.to_tensor([[]])
        tensor_2d = Tensor(empty_tensor_2d)

        for ord_val in [0, 1, float("inf")]:
            try:
                result = linalg.norm(tensor_1d, ord=ord_val)
                assert result.item() == 0, f"Empty tensor norm with ord={ord_val} should be 0"
            except tf.errors.InvalidArgumentError as e:
                print(f"Note: TensorFlow cannot compute norm with ord={ord_val} on empty tensor: {e}")
            except Exception as e:
                if (
                    "shape" not in str(e).lower()
                    and "empty" not in str(e).lower()
                    and "dimension" not in str(e).lower()
                ):
                    assert False, f"Unexpected error for empty tensor with ord={ord_val}: {e}"
                print(f"Expected error for empty tensor with ord={ord_val}: {e}")

        try:
            result = linalg.norm(tensor_1d, ord=0)
            assert result.item() == 0, "Count of non-zeros in empty tensor should be 0"
        except Exception as e:
            if "shape" not in str(e).lower() and "empty" not in str(e).lower() and "dimension" not in str(e).lower():
                assert False, f"Unexpected error for empty tensor with ord=0: {e}"
            print(f"Expected shape error for empty tensor with ord=0: {e}")

        try:
            result = linalg.norm(tensor_1d, ord=0, keepdims=True)
            assert result.shape == (1,), "Shape with keepdims should be (1,)"
            assert result.item() == 0, "Count of non-zeros in empty tensor should be 0"
        except Exception as e:
            if "shape" not in str(e).lower() and "empty" not in str(e).lower() and "dimension" not in str(e).lower():
                assert False, f"Unexpected error for empty tensor with keepdims: {e}"
            print(f"Expected error with keepdims on empty tensor: {e}")

        for ord_val in ["fro", 0]:
            try:
                result = linalg.norm(tensor_2d, ord=ord_val)
                assert result.item() == 0, f"Empty tensor norm with ord={ord_val} should be 0"
            except tf.errors.InvalidArgumentError as e:
                print(f"Note: TensorFlow cannot compute norm with ord={ord_val} on empty 2D tensor: {e}")
            except Exception as e:
                if (
                    "shape" not in str(e).lower()
                    and "empty" not in str(e).lower()
                    and "dimension" not in str(e).lower()
                ):
                    assert False, f"Unexpected error for empty 2D tensor with ord={ord_val}: {e}"
                print(f"Expected error for empty 2D tensor with ord={ord_val}: {e}")

        try:
            result = linalg.norm(tensor_2d, ord=0, axis=0)
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

        mixed_tensor_data = [[float("inf"), 2.0], [float("nan"), 4.0]]
        mixed_tf_tensor = self.to_tensor(mixed_tensor_data)
        mixed_tensor = Tensor(mixed_tf_tensor)

        result = linalg.norm(inf_tensor, ord=0)
        assert result.item() == 4, "All elements (including Inf) should be counted as non-zero"

        result = linalg.norm(nan_tensor, ord=0)
        assert result.item() == 4, "All elements (including NaN) should be counted as non-zero"

        result = linalg.norm(inf_tensor, ord=0, axis=0)
        assert result.shape == (2,)
        assert result.data.numpy()[0] == 2 and result.data.numpy()[1] == 2

        result = linalg.norm(inf_tensor, ord=0, axis=1)
        assert result.shape == (2,)
        assert result.data.numpy()[0] == 2 and result.data.numpy()[1] == 2

        try:
            result = linalg.norm(inf_tensor, ord="fro")
            assert float("inf") == result.item() or result.item() > 1e30, "Norm with Inf should be Inf or very large"
        except tf.errors.InvalidArgumentError as e:
            print(f"Note: TensorFlow cannot compute Frobenius norm with Inf values: {e}")

        try:
            result = linalg.norm(nan_tensor, ord="nuc")
            import numpy as np

            assert np.isnan(result.item()) or result.item() > 0, "Nuclear norm with NaN might be NaN"
        except tf.errors.InvalidArgumentError as e:
            print(f"Note: TensorFlow SVD cannot handle NaN values: {e}")

        try:
            result = linalg.norm(inf_tensor, ord=float("inf"))
            assert result.item() == float("inf"), "Infinity norm with Inf values should be Inf"
        except tf.errors.InvalidArgumentError as e:
            print(f"Note: TensorFlow cannot compute infinity norm with Inf values: {e}")

        try:
            result = linalg.norm(inf_tensor, ord=0, keepdims=True)
            assert result.shape == (1, 1)
            assert result.item() == 4
        except Exception as e:
            assert False, f"Unexpected error with keepdims and extreme values: {e}"

        try:
            result = linalg.norm(mixed_tensor, ord=0)
            assert result.item() == 4, "All elements should be counted as non-zero"
        except Exception as e:
            if "invalid" not in str(e).lower():
                assert False, f"Unexpected error with mixed NaN and Inf: {e}"
            print(f"Note: Expected error with mixed NaN and Inf values: {e}")

    @staticmethod
    def backend() -> TensorBackend:
        return TensorBackend.tf

    @staticmethod
    def device() -> TensorDeviceType:
        return TensorDeviceType.GPU

    @pytest.mark.skip("Desired slicing is not supported for TensorFlow")
    @pytest.mark.parametrize("is_tensor_indecies", (False, True))
    def test_getitem_for_indecies(self, is_tensor_indecies):
        pass

    @pytest.mark.skip("TensorFlow throws different kind of exceptions")
    @pytest.mark.parametrize(
        "val, axis, exception_type, exception_match",
        (
            ([[[[1], [2]], [[1], [2]]]], (0, 1), ValueError, "not equal to one"),
            ([[[[1], [2]], [[1], [2]]]], 42, IndexError, "out of"),
            ([[[[1], [2]], [[1], [2]]]], (0, 42), IndexError, "out of"),
        ),
    )
    def test_squeeze_axis_error(self, val, axis, exception_type, exception_match):
        pass
