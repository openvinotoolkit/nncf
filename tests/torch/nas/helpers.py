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

import torch
import torch.nn.functional as F

from nncf.torch.graph.graph_builder import GraphConverter


def do_conv2d(conv, input_, *, padding=None, weight=None, bias=None):
    weight = conv.weight if weight is None else weight
    bias = conv.bias if bias is None else bias
    padding = conv.padding if padding is None else padding
    return F.conv2d(input_, weight, bias, conv.stride, padding, conv.dilation, conv.groups)


def do_training_step(model, optimizer, input_):
    model.train()
    output = model(input_)
    optimizer.zero_grad()
    output.sum().backward()
    optimizer.step()
    output = model(input_)
    return output


def compare_tensors_ignoring_the_order(t1: torch.Tensor, t2: torch.Tensor, rtol=1e-05, atol=1e-08):
    t1 = torch.flatten(t1)
    t2 = torch.flatten(t2)
    assert torch.allclose(torch.sort(t1)[0], torch.sort(t2)[0], rtol=rtol, atol=atol)


def ref_kernel_transform(weights, target_kernel_size=3, start=1, end=4, transition_matrix=None):
    if transition_matrix is None:
        transition_matrix = torch.eye(target_kernel_size**2)
    weights = weights[:, :, start:end, start:end]
    out_channels = weights.size(0)
    in_channels = weights.size(1)
    weights = weights.reshape(weights.size(0), weights.size(1), -1)
    weights = weights.view(-1, weights.size(2))
    weights = F.linear(weights, transition_matrix)
    weights = weights.view(out_channels, in_channels, target_kernel_size**2)
    weights = weights.view(out_channels, in_channels, target_kernel_size, target_kernel_size)
    return weights


def move_model_to_cuda_if_available(model):
    if torch.cuda.is_available():
        model.cuda()
    return next(iter(model.parameters())).device


class DebugGraphContext:
    def __init__(self, model, input_info):
        self._model = model
        self._input_info = input_info

    def dump_graph(self, file_name):
        dyn_graph = self._model.nncf.get_dynamic_graph()
        nncf_graph = GraphConverter.convert(dyn_graph, False)
        nncf_graph.visualize_graph(f"{file_name}.dot")

    def __enter__(self):
        self.dump_graph("before")
        return self

    def __exit__(self, *args):
        self.dump_graph("after")
