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

from nncf.tensorflow.pruning.tensor_processor import TFNNCFPruningTensorProcessor
from nncf.tensorflow.tensor import TFNNCFTensor


@pytest.mark.parametrize("device", ("CPU", pytest.param("GPU", marks=pytest.mark.nightly)))
def test_create_tensor(device):
    if not tf.config.list_physical_devices("GPU") and device == "GPU":
        pytest.skip("There are no available CUDA devices")
    shape = [1, 3, 10, 100]
    tensor = TFNNCFPruningTensorProcessor.ones(shape, device)
    assert tf.is_tensor(tensor.tensor)
    assert tensor.tensor.device.split("/")[-1].split(":")[1] == device
    assert list(tensor.tensor.shape) == shape


def test_repeat():
    tensor_data = [0.0, 1.0]
    repeats = 5
    tensor = TFNNCFTensor(tf.Variable(tensor_data))
    repeated_tensor = TFNNCFPruningTensorProcessor.repeat(tensor, repeats=repeats)
    ref_repeated = []
    for val in tensor_data:
        for _ in range(repeats):
            ref_repeated.append(val)
    assert tf.reduce_all(repeated_tensor.tensor == tf.Variable(ref_repeated))


def test_concat():
    tensor_data = [0.0, 1.0]
    tensors = [TFNNCFTensor(tf.Variable(tensor_data)) for _ in range(3)]
    concatenated_tensor = TFNNCFPruningTensorProcessor.concatenate(tensors, axis=0)
    assert tf.reduce_all(concatenated_tensor.tensor == tf.Variable(tensor_data * 3))


@pytest.mark.parametrize("all_close", [False, True])
def test_assert_all_close(all_close):
    tensor_data = [0.0, 1.0]
    tensors = [TFNNCFTensor(tf.Variable(tensor_data)) for _ in range(3)]
    if not all_close:
        tensors.append(TFNNCFTensor(tf.Variable(tensor_data[::-1])))
        with pytest.raises(tf.errors.InvalidArgumentError):
            TFNNCFPruningTensorProcessor.assert_allclose(tensors)
    else:
        TFNNCFPruningTensorProcessor.assert_allclose(tensors)


@pytest.mark.parametrize("all_close", [False, True])
def test_elementwise_mask_propagation(all_close):
    tensor_data = [0.0, 1.0]
    tensors = [TFNNCFTensor(tf.Variable(tensor_data)) for _ in range(3)]
    if not all_close:
        tensors.append(TFNNCFTensor(tf.Variable(tensor_data[::-1])))
        with pytest.raises(tf.errors.InvalidArgumentError):
            TFNNCFPruningTensorProcessor.elementwise_mask_propagation(tensors)
    else:
        result = TFNNCFPruningTensorProcessor.elementwise_mask_propagation(tensors)
        for t in tensors:
            tf.debugging.assert_near(result.tensor, t.tensor)


def test_split():
    tensor_data = [0.0, 1.0, 2.0, 3.0]
    tf_variable = tf.Variable(tensor_data)
    tf_output = tf.split(tf_variable, 2)
    output_shapes = [output.shape[0] for output in tf_output]
    tensor = TFNNCFTensor(tf.Variable(tensor_data))
    split_tensors = TFNNCFPruningTensorProcessor.split(tensor, output_shapes=output_shapes)
    assert tf.reduce_all(split_tensors[0].tensor == tensor_data[:2])
    assert tf.reduce_all(split_tensors[1].tensor == tensor_data[2:])
