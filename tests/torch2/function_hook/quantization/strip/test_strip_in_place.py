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

from dataclasses import dataclass
from itertools import product
from typing import Any

import pytest
import torch
from torch import nn

import nncf
from nncf.parameters import CompressionFormat
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import BaseQuantizer
from tests.torch.helpers import LinearModel
from tests.torch2.function_hook.quantization.strip.test_strip_dequantize import check_compression_modules


@dataclass
class ParamInPlaceStrip:
    mode: CompressWeightsMode
    compression_format: CompressionFormat
    torch_dtype: torch.dtype

    def __str__(self) -> str:
        return f"{self.mode}_{self.torch_dtype}_{self.compression_format}"

    @property
    def extra_arguments(self) -> dict[str, Any]:
        args = {}
        if self.mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
            args = {"ratio": 1, "group_size": 4, "all_layers": True}
        if self.compression_format == CompressionFormat.FQ:
            args["group_size"] = -1
        return args


@pytest.mark.parametrize(
    "param",
    [
        ParamInPlaceStrip(mode, compression_format, torch_dtype)
        for mode, compression_format, torch_dtype in product(
            [
                CompressWeightsMode.INT4_ASYM,
                CompressWeightsMode.INT4_SYM,
                CompressWeightsMode.INT8_ASYM,
                CompressWeightsMode.INT8_SYM,
            ],
            [CompressionFormat.FQ_LORA, CompressionFormat.FQ],
            [torch.float32, torch.float16, torch.bfloat16],
        )
    ],
    ids=str,
)
def test_nncf_in_place_strip(param: ParamInPlaceStrip):
    input_shape = [1, 16]
    model = LinearModel(input_shape=input_shape).to(param.torch_dtype)
    example_input = torch.ones(input_shape).to(param.torch_dtype)

    compressed_model = nncf.compress_weights(
        model,
        mode=param.mode,
        dataset=nncf.Dataset([example_input]),
        compression_format=param.compression_format,
        advanced_parameters=nncf.AdvancedCompressionParameters(lora_adapter_rank=1),
        **param.extra_arguments,
    )

    check_compression_modules(compressed_model, expected_class=BaseQuantizer)
    assert compressed_model.linear.weight.dtype == param.torch_dtype

    with torch.no_grad():
        compressed_output = compressed_model(example_input)
        strip_compressed_model = nncf.strip(
            compressed_model, do_copy=False, strip_format=StripFormat.IN_PLACE, example_input=example_input
        )
        stripped_output = strip_compressed_model(example_input)

        assert strip_compressed_model.linear.weight.dtype == param.torch_dtype
        hook_storage = get_hook_storage(strip_compressed_model)
        assert not list(hook_storage.named_hooks())
        assert torch.allclose(compressed_output, stripped_output)


def test_nncf_in_place_strip_keeps_other_hooks():
    input_shape = [1, 16]
    model = LinearModel(input_shape=input_shape)
    example_input = torch.ones(input_shape)

    compressed_model = nncf.compress_weights(
        model,
        mode=CompressWeightsMode.INT8_ASYM,
        group_size=-1,
        dataset=nncf.Dataset([example_input]),
        compression_format=CompressionFormat.FQ,
    )

    hook1 = nn.Identity()
    hook_storage = get_hook_storage(compressed_model)
    hook_storage.register_post_function_hook("linear:weight", 0, hook1)
    ret = [name for name, _ in hook_storage.named_hooks(remove_duplicate=False)]
    ref = ["post_hooks.linear:weight__0.0", "post_hooks.linear:weight__0.1"]
    assert ret == ref

    model = nncf.strip(model, do_copy=False, strip_format=StripFormat.IN_PLACE, example_input=example_input)
    ret = list(hook_storage.named_hooks(remove_duplicate=False))
    ref = [("post_hooks.linear:weight__0.1", hook1)]
    assert ret == ref
