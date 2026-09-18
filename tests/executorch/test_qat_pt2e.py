# Copyright (c) 2026 Intel Corporation
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
from torchao.quantization.pt2e import move_exported_model_to_eval
from torchao.quantization.pt2e import move_exported_model_to_train
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import get_default_x86_inductor_quantization_config

from nncf.experimental.torch.fx import quantize_qat_pt2e
from tests.torch.fx.helpers import get_torch_fx_model


class ConvReluModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


def test_quantize_qat_pt2e():
    example_input = torch.randn(2, 3, 16, 16)

    model = ConvReluModel()
    fx_model = get_torch_fx_model(model, example_input)

    quantizer = X86InductorQuantizer()
    quantizer.set_global(get_default_x86_inductor_quantization_config(is_qat=True))

    qat_model = quantize_qat_pt2e(
        fx_model,
        quantizer,
    )

    assert any(
        node.op == "call_module" and str(node.target).startswith("activation_post_process_")
        for node in qat_model.graph.nodes
    )

    move_exported_model_to_train(qat_model)

    optimizer = torch.optim.SGD(qat_model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    output = qat_model(example_input)
    loss = output.mean()
    loss.backward()
    optimizer.step()

    assert loss.isfinite()

    move_exported_model_to_eval(qat_model)
    quantized_model = convert_pt2e(qat_model)

    graph = str(quantized_model.graph)

    assert "quantized_decomposed.quantize" in graph
    assert "quantized_decomposed.dequantize" in graph
