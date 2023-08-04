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

import pytest
import tensorflow as tf

from nncf.experimental.tensor import Tensor
from nncf.tensorflow.pruning.tensor_processor import TFNNCFPruningTensorProcessor


@pytest.mark.parametrize("device", ("CPU", "GPU"))
def test_create_tensor(device):
    if not tf.config.list_physical_devices("GPU"):
        if device == "GPU":
            pytest.skip("There are no available CUDA devices")
    shape = [1, 3, 10, 100]
    tensor = TFNNCFPruningTensorProcessor.ones(shape, device)
    assert tf.is_tensor(tensor.data)
    assert tensor.data.device.split("/")[-1].split(":")[1] == device
    assert list(tensor.data.shape) == shape


def test_repeat():
    tensor_data = [0.0, 1.0]
    repeats = 5
    tensor = Tensor(tf.Variable(tensor_data))
    repeated_tensor = TFNNCFPruningTensorProcessor.repeat(tensor, repeats=repeats)
    ref_repeated = []
    for val in tensor_data:
        for _ in range(repeats):
            ref_repeated.append(val)
    assert tf.reduce_all(repeated_tensor.data == tf.Variable(ref_repeated))


def test_concat():
    tensor_data = [0.0, 1.0]
    tensors = [Tensor(tf.Variable(tensor_data)) for _ in range(3)]
    concatenated_tensor = TFNNCFPruningTensorProcessor.concatenate(tensors, axis=0)
    assert tf.reduce_all(concatenated_tensor.data == tf.Variable(tensor_data * 3))


@pytest.mark.parametrize("all_close", [False, True])
def test_assert_all_close(all_close):
    tensor_data = [0.0, 1.0]
    tensors = [Tensor(tf.Variable(tensor_data)) for _ in range(3)]
    if not all_close:
        tensors.append(Tensor(tf.Variable(tensor_data[::-1])))
        with pytest.raises(tf.errors.InvalidArgumentError):
            TFNNCFPruningTensorProcessor.assert_allclose(tensors)
    else:
        TFNNCFPruningTensorProcessor.assert_allclose(tensors)


@pytest.mark.parametrize("all_close", [False, True])
def test_elementwise_mask_propagation(all_close):
    tensor_data = [0.0, 1.0]
    tensors = [Tensor(tf.Variable(tensor_data)) for _ in range(3)]
    if not all_close:
        tensors.append(Tensor(tf.Variable(tensor_data[::-1])))
        with pytest.raises(tf.errors.InvalidArgumentError):
            TFNNCFPruningTensorProcessor.elementwise_mask_propagation(tensors)
    else:
        result = TFNNCFPruningTensorProcessor.elementwise_mask_propagation(tensors)
        for t in tensors:
            tf.debugging.assert_near(result.data, t.data)


def test_split():
    tensor_data = [0.0, 1.0, 2.0, 3.0]
    tf_variable = tf.Variable(tensor_data)
    tf_output = tf.split(tf_variable, 2)
    output_shapes = [output.shape[0] for output in tf_output]
    tensor = Tensor(tf.Variable(tensor_data))
    split_tensors = TFNNCFPruningTensorProcessor.split(tensor, output_shapes=output_shapes)
    assert tf.reduce_all(split_tensors[0].data == tensor_data[:2])
    assert tf.reduce_all(split_tensors[1].data == tensor_data[2:])
