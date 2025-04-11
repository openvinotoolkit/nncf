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
