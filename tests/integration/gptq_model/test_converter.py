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

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import nncf
from nncf.experimental.torch.gptqmodel.convertor import convert_linear_module
from nncf.experimental.torch.gptqmodel.convertor import convert_model
from nncf.torch.function_hook.wrapper import ForwardWithHooks
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor


@torch.no_grad()
def print_tensor(t: torch.Tensor):
    print()
    print("=" * 80)
    print(f"values = {t.flatten()[:8].tolist()}")
    print(f"type   = {t.dtype}, shape={tuple(t.shape)}")
    print(f"max    = {t.max()}, min={t.min()}, mean={t.mean()}, std={t.std()}")
    print("=" * 80)


@torch.no_grad()
@pytest.mark.parametrize("with_bias", [True, False], ids=["with_bias", "without_bias"])
@pytest.mark.parametrize("is_sym", [True, False], ids=["sym", "asym"])
def test_convert_int8_linear(with_bias: bool, is_sym: bool):
    torch.manual_seed(42)

    example_input = torch.randn(1, 128, dtype=torch.float16)

    # Prepare nncf style compressed weights and decompressor
    scale = torch.rand((128, 1), dtype=torch.float16) * 0.01
    compressed_weight = torch.randint(0, 255, (128, 128), dtype=torch.uint8)
    bias = torch.randn((128,), dtype=torch.float16) if with_bias else None

    if is_sym:
        decompressor = INT8SymmetricWeightsDecompressor(scale=scale)
    else:
        zp = torch.randint(100, 140, (128, 1), dtype=torch.uint8)
        decompressor = INT8AsymmetricWeightsDecompressor(scale=scale, zero_point=zp)

    result = F.linear(example_input, decompressor(compressed_weight), bias)

    # Build nn.Linear after weight_compression
    linear = nn.Linear(128, 128, bias=with_bias)
    linear.weight = nn.Parameter(compressed_weight, requires_grad=False)
    if with_bias:
        linear.bias = nn.Parameter(bias, requires_grad=False)

    # Convert to gtpqmodel module
    gptq_module = convert_linear_module(linear, decompressor)
    gptq_module.cuda()
    gptq_result = gptq_module(example_input.cuda())
    print_tensor(gptq_result)

    torch.testing.assert_close(result, gptq_result.cpu(), atol=1e-3, rtol=1e-3)


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(128, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def test_convert_model():
    example_input = torch.randn(1, 128).cuda()
    model = LinearModel().cuda()

    assert isinstance(model.linear, nn.Linear)
    q_model = nncf.compress_weights(
        model, mode=nncf.CompressWeightsMode.INT8_SYM, dataset=nncf.Dataset([example_input])
    )

    convert_model(q_model)

    # Linear layers have been converted
    assert not isinstance(q_model.linear, nn.Linear)
    # Model was unwrapped
    assert not isinstance(q_model.forward, ForwardWithHooks)
