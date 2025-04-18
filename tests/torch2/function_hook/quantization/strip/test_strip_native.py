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

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.quantization import FakeQuantize

import nncf
import nncf.torch
from nncf.quantization.advanced_parameters import OverflowFix
from nncf.torch.function_hook.wrapper import get_hook_storage
from tests.common.quantization.data_generators import generate_lazy_sweep_data
from tests.torch.helpers import BasicConvTestModel


@pytest.mark.parametrize("strip_type", ("nncf", "torch"))
@pytest.mark.parametrize("do_copy", (True, False), ids=["copy", "inplace"])
def test_nncf_strip_api(strip_type: str, do_copy: bool):
    model = BasicConvTestModel()
    example_input = torch.ones(model.INPUT_SIZE)
    quantized_model = nncf.quantize(model, nncf.Dataset([torch.ones(model.INPUT_SIZE)]), subset_size=1)

    if strip_type == "nncf":
        strip_model = nncf.strip(quantized_model, do_copy, nncf.StripFormat.NATIVE, example_input)
    elif strip_type == "torch":
        strip_model = nncf.torch.strip(quantized_model, do_copy, nncf.StripFormat.NATIVE, example_input)

    if do_copy:
        assert id(strip_model) != id(quantized_model)
    else:
        assert id(strip_model) == id(quantized_model)

    num_fq = 0
    for name, hook in get_hook_storage(strip_model).named_hooks():
        assert isinstance(hook, FakeQuantize), f"{name} is {type(hook)} but expected FakeQuantize"
        num_fq += 1

    assert num_fq == 2


def check_quantizer_operators(model: nn.Module, levels: int = 255):
    """Check that model contains only 8bit FakeQuantize operators."""
    hook_storage = get_hook_storage(model)
    for _, hook in hook_storage.named_hooks():
        assert isinstance(hook, FakeQuantize)
        assert hook.quant_max - hook.quant_min == levels


@pytest.mark.parametrize("overflow_fix", (OverflowFix.DISABLE, OverflowFix.ENABLE))
def test_strip_quantization(overflow_fix: OverflowFix, tmp_path: Path):
    model = BasicConvTestModel()
    example_input = torch.tensor(generate_lazy_sweep_data(model.INPUT_SIZE), dtype=torch.float32)
    q_model = nncf.quantize(
        model,
        nncf.Dataset([example_input]),
        advanced_parameters=nncf.AdvancedQuantizationParameters(overflow_fix=overflow_fix),
    )
    input_tensor = torch.Tensor(generate_lazy_sweep_data(model.INPUT_SIZE))

    with torch.no_grad():
        x_nncf = q_model(input_tensor)

    inference_model = nncf.strip(q_model, example_input=example_input)
    with torch.no_grad():
        x_torch = inference_model(input_tensor)

    check_quantizer_operators(inference_model, 2**8 - 1)
    assert torch.all(torch.isclose(x_nncf, x_torch)), f"{x_nncf.view(-1)} != {x_torch.view(-1)}"

    torch.onnx.export(inference_model, input_tensor, f"{tmp_path}/model.onnx")
