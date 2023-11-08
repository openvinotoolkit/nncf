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

import torch
from torch import nn

from nncf.data import Dataset
from nncf.torch.quantization.quantize_model import create_nncf_network_ptq


class TestModel(nn.Module):
    def __init__(self, input_shape) -> None:
        super().__init__()
        self._input_shape = input_shape

    def forward(self, x):
        assert x.shape == self._input_shape
        return x


def test_create_nncf_network_with_nncf_dataset():
    input_shape = (1, 3, 5, 5)
    model = TestModel(input_shape)

    def transform_fn(inputs):
        x, _ = inputs
        return x

    dataset = Dataset([(torch.empty(input_shape), 1)] * 3, transform_fn)
    nncf_network = create_nncf_network_ptq(model, dataset)
    nncf_graph = nncf_network.nncf.get_original_graph()
    all_nodes = nncf_graph.get_all_nodes()
    assert len(all_nodes) == 2
    assert sorted([node.node_type for node in all_nodes]) == ["nncf_model_input", "nncf_model_output"]
