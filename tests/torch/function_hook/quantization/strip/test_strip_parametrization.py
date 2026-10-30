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
from torch import nn
from torch.nn.utils import parametrize

import nncf
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.function_hook.wrapper import wrap_model
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from tests.torch.helpers import LinearModel

INPUT_SHAPE = [1, 16]


class TiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 16)
        self.head = nn.Linear(16, 16, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))


def _compress(model: nn.Module, example_input: torch.Tensor, mode: CompressWeightsMode) -> nn.Module:
    extra_args = {"ratio": 1, "group_size": 4, "all_layers": True} if mode.value.startswith("int4") else {}
    return nncf.compress_weights(
        model,
        mode=mode,
        dataset=nncf.Dataset([example_input]),
        compression_format=nncf.CompressionFormat.FQ_LORA,
        advanced_parameters=nncf.AdvancedCompressionParameters(lora_adapter_rank=1),
        **extra_args,
    )


@pytest.mark.parametrize(
    "mode",
    (
        CompressWeightsMode.INT4_SYM,
        CompressWeightsMode.INT4_ASYM,
        CompressWeightsMode.INT8_SYM,
        CompressWeightsMode.INT8_ASYM,
    ),
    ids=lambda mode: mode.value,
)
def test_strip_rp_linear(mode: CompressWeightsMode) -> None:
    example_input = torch.ones(INPUT_SHAPE)
    compressed_model = _compress(LinearModel(input_shape=INPUT_SHAPE), example_input, mode)

    with torch.no_grad():
        ref_output = compressed_model(example_input)
        stripped_model = nncf.strip(compressed_model, do_copy=True, strip_format=StripFormat.RP)
        output = stripped_model(example_input)

    # Quantizers are removed from the hook storage and replaced by a parametrization
    assert not list(get_hook_storage(stripped_model).named_hooks())
    assert parametrize.is_parametrized(stripped_model.linear, "weight")
    decompressors = stripped_model.linear.parametrizations.weight
    assert len(decompressors) == 1
    assert isinstance(decompressors[0], BaseWeightsDecompressor)

    original_weight = stripped_model.linear.parametrizations.weight.original
    assert original_weight.dtype in (torch.uint8, torch.int8)
    assert not original_weight.requires_grad
    assert torch.allclose(ref_output, output, atol=1e-3)


def test_strip_rp_in_place() -> None:
    example_input = torch.ones(INPUT_SHAPE)
    compressed_model = _compress(LinearModel(input_shape=INPUT_SHAPE), example_input, CompressWeightsMode.INT8_SYM)

    stripped_model = nncf.strip(compressed_model, do_copy=False, strip_format=StripFormat.RP)

    assert stripped_model is compressed_model


def test_strip_rp_tied_weights() -> None:
    example_input = torch.arange(4)
    model = TiedModel()
    compressed_model = _compress(model, example_input, CompressWeightsMode.INT8_SYM)
    with torch.no_grad():
        ref_output = compressed_model(example_input)
        stripped_model = nncf.strip(compressed_model, do_copy=False, strip_format=StripFormat.RP)
        output = stripped_model(example_input)

    # All modules sharing the packed weight must decompress it
    for module in (stripped_model.embedding, stripped_model.head):
        assert parametrize.is_parametrized(module, "weight")
    assert torch.allclose(ref_output, output, atol=5e-2)


def test_strip_rp_without_compression_hooks() -> None:
    model = LinearModel(input_shape=INPUT_SHAPE)
    ref_weight = model.linear.weight.clone()

    stripped_model = nncf.strip(wrap_model(model), strip_format=StripFormat.RP)

    assert not parametrize.is_parametrized(stripped_model.linear)
    assert torch.equal(stripped_model.linear.weight, ref_weight)
