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
from torch import nn

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import AsymmetricQuantizer
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricQuantizer


class SimpleModel(nn.Module):
    def __init__(self, fq) -> None:
        super().__init__()
        self.fq = fq

    def forward(self, x):
        return self.fq(x)


def check_fq_op(traced_graph: nn.Module, is_per_channel: bool):
    aten_op = "aten::fake_quantize_per_channel_affine" if is_per_channel else "aten::fake_quantize_per_tensor_affine"
    is_fq_node = False
    for graph_node in traced_graph.inlined_graph.nodes():
        if graph_node.kind() == "prim::PythonOp" and "Subgraph" in graph_node.attributeNames():
            subgraph = getattr(graph_node, graph_node.kindOf("Subgraph"))("Subgraph")
            for subgraph_node in subgraph.nodes():
                if subgraph_node.kind() == aten_op:
                    is_fq_node = True
                    break
        if is_fq_node:
            break

    assert is_fq_node, "FQ operation is not found in the traced graph"


def test_trace_asymmetric_quantizer(is_per_channel):
    if is_per_channel:
        input_low = torch.tensor([-0.1, 0.1]).reshape(1, 2, 1, 1)
        input_range = torch.tensor([0.3, 0.4]).reshape(1, 2, 1, 1)
    else:
        input_low = torch.tensor([-0.1])
        input_range = torch.tensor([1.1])

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.ASYMMETRIC,
        signedness_to_force=False,
        narrow_range=False,
        scale_shape=tuple(input_low.shape),
        logarithm_scale=False,
        half_range=False,
        is_quantized_on_export=True,
    )
    quantizer = AsymmetricQuantizer(qspec)
    quantizer.input_low.data = input_low
    quantizer.input_range.data = input_range

    model = SimpleModel(quantizer)
    traced = torch.jit.trace(model, torch.ones(1, 2, 1, 1))
    check_fq_op(traced, is_per_channel)


def test_trace_symmetric_quantizer(is_per_channel, is_signed):
    if is_per_channel:
        scale = torch.tensor([0.3, 0.4]).reshape(1, 2, 1, 1)
    else:
        scale = torch.tensor([1.1])

    qspec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=False,
        narrow_range=False,
        scale_shape=tuple(scale.shape),
        logarithm_scale=False,
        half_range=False,
        is_quantized_on_export=True,
    )
    quantizer = SymmetricQuantizer(qspec)
    quantizer.scale.data = scale
    quantizer.signed = is_signed

    model = SimpleModel(quantizer)
    traced = torch.jit.trace(model, torch.ones(1, 2, 1, 1))
    check_fq_op(traced, is_per_channel)
