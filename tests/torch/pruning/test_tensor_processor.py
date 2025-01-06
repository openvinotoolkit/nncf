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
import torch

from nncf.torch.pruning.tensor_processor import PTNNCFPruningTensorProcessor
from nncf.torch.tensor import PTNNCFTensor


def test_ones(use_cuda):
    if use_cuda and not torch.cuda.is_available():
        pytest.skip("There are no available CUDA devices")
    device = torch.device("cuda" if use_cuda else "cpu")
    shape = [1, 3, 10, 100]
    tensor = PTNNCFPruningTensorProcessor.ones(shape, device)
    assert torch.is_tensor(tensor.tensor)
    assert tensor.device.type == device.type
    assert list(tensor.tensor.shape) == shape


def test_repeat():
    tensor_data = [0.0, 1.0]
    repeats = 5
    tensor = PTNNCFTensor(torch.tensor(tensor_data))
    repeated_tensor = PTNNCFPruningTensorProcessor.repeat(tensor, repeats=repeats)
    ref_repeated = []
    for val in tensor_data:
        for _ in range(repeats):
            ref_repeated.append(val)
    assert torch.all(repeated_tensor.tensor == torch.tensor(ref_repeated))


def test_concat():
    tensor_data = [0.0, 1.0]
    tensors = [PTNNCFTensor(torch.tensor(tensor_data)) for _ in range(3)]
    concatenated_tensor = PTNNCFPruningTensorProcessor.concatenate(tensors, axis=0)
    assert torch.all(concatenated_tensor.tensor == torch.tensor(tensor_data * 3))


@pytest.mark.parametrize("all_close", [False, True])
def test_assert_all_close(all_close):
    tensor_data = [0.0, 1.0]
    tensors = [PTNNCFTensor(torch.tensor(tensor_data)) for _ in range(3)]
    if not all_close:
        tensors.append(PTNNCFTensor(torch.tensor(tensor_data[::-1])))
        with pytest.raises(AssertionError):
            PTNNCFPruningTensorProcessor.assert_allclose(tensors)
    else:
        PTNNCFPruningTensorProcessor.assert_allclose(tensors)


@pytest.mark.parametrize("all_close", [False, True])
def test_elementwise_mask_propagation(all_close):
    tensor_data = [0.0, 1.0]
    tensors = [PTNNCFTensor(torch.tensor(tensor_data)) for _ in range(3)]
    if not all_close:
        tensors.append(PTNNCFTensor(torch.tensor(tensor_data[::-1])))
        with pytest.raises(AssertionError):
            PTNNCFPruningTensorProcessor.elementwise_mask_propagation(tensors)
    else:
        result = PTNNCFPruningTensorProcessor.elementwise_mask_propagation(tensors)
        for t in tensors:
            assert torch.allclose(result.tensor, t.tensor)


def test_split():
    tensor_data = [0.0, 1.0, 2.0, 3.0]
    chunks = 2
    pt_tensor = torch.tensor(tensor_data)
    pt_output = torch.chunk(pt_tensor, chunks=2)
    output_shapes = [output.shape[0] for output in pt_output]
    tensor = PTNNCFTensor(pt_tensor)
    split_tensors = PTNNCFPruningTensorProcessor.split(tensor, output_shapes=output_shapes)
    ref_split = torch.tensor(tensor_data).chunk(chunks)
    assert torch.all(split_tensors[0].tensor == ref_split[0])
    assert torch.all(split_tensors[1].tensor == ref_split[1])
