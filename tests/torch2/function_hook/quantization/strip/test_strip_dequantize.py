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
from typing import Any

import openvino as ov
import pytest
import torch
from openvino._pyopenvino.properties.hint import inference_precision
from openvino.tools.ovc import convert_model
from pytest_mock import MockerFixture
from torch import nn

import nncf
import nncf.torch
from nncf.common.quantization.structs import QuantizationScheme
from nncf.openvino.optimized_functions.models import _compile_ov_model
from nncf.parameters import CompressWeightsMode
from nncf.parameters import StripFormat
from nncf.torch.function_hook.wrapper import get_hook_storage
from nncf.torch.quantization.layers import AsymmetricLoraQuantizer
from nncf.torch.quantization.layers import BaseQuantizer
from nncf.torch.quantization.layers import BaseWeightsDecompressor
from nncf.torch.quantization.layers import INT4AsymmetricWeightsDecompressor as INT4AsymDQ
from nncf.torch.quantization.layers import INT4SymmetricWeightsDecompressor as INT4SymDQ
from nncf.torch.quantization.layers import INT8AsymmetricWeightsDecompressor as INT8AsymDQ
from nncf.torch.quantization.layers import INT8SymmetricWeightsDecompressor as INT8SymDQ
from nncf.torch.quantization.layers import PTLoraSpec
from nncf.torch.quantization.layers import PTQuantizerSpec
from nncf.torch.quantization.layers import SymmetricLoraQuantizer
from nncf.torch.quantization.strip import asym_fq_to_decompressor
from nncf.torch.quantization.strip import sym_fq_to_decompressor
from tests.torch2.helpers import LinearModel


def check_compression_modules(
    model: nn.Module,
    expected_class: type,
) -> None:
    hook_storage = get_hook_storage(model)
    hooks = list(hook_storage.named_hooks())
    assert len(hooks) == 1
    for _, module in hooks:
        assert isinstance(module, expected_class)


@dataclass
class ParamStripLora:
    mode: CompressWeightsMode
    decompressor_class: type
    torch_dtype: torch.dtype
    torch_atol: float
    ov_atol: float
    weight_dtype: torch.dtype

    def __str__(self) -> str:
        return f"{self.mode}_{self.torch_dtype}"

    @property
    def extra_arguments(self) -> dict[str, Any]:
        if self.mode in [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT4_ASYM]:
            return {"ratio": 1, "group_size": 4, "all_layers": True}
        return {}

    @property
    def num_call_pack_weight(self) -> int:
        if self.mode in [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM]:
            # pack_weight for asym is called twice: for ZP and weight
            return 2
        return 1


@pytest.mark.parametrize(
    ("param"),
    (
        ParamStripLora(CompressWeightsMode.INT4_ASYM, INT4AsymDQ, torch.float32, 1e-3, 1e-3, torch.uint8),
        ParamStripLora(CompressWeightsMode.INT4_ASYM, INT4AsymDQ, torch.float16, 1e-3, 1e-3, torch.uint8),
        ParamStripLora(CompressWeightsMode.INT4_ASYM, INT4AsymDQ, torch.bfloat16, 1e-8, 1e-1, torch.uint8),
        ParamStripLora(CompressWeightsMode.INT4_SYM, INT4SymDQ, torch.float32, 1e-3, 1e-3, torch.uint8),
        ParamStripLora(CompressWeightsMode.INT4_SYM, INT4SymDQ, torch.float16, 1e-8, 1e-3, torch.uint8),
        ParamStripLora(CompressWeightsMode.INT4_SYM, INT4SymDQ, torch.bfloat16, 1e-8, 1e-2, torch.uint8),
        ParamStripLora(CompressWeightsMode.INT8_SYM, INT8SymDQ, torch.bfloat16, 1e-2, 1e-3, torch.int8),
        ParamStripLora(CompressWeightsMode.INT8_ASYM, INT8AsymDQ, torch.bfloat16, 1e-8, 1e-3, torch.uint8),
    ),
    ids=str,
)
def test_nncf_strip_lora_model(param: ParamStripLora, mocker: MockerFixture):
    input_shape = [1, 16]
    model = LinearModel(input_shape=input_shape).to(param.torch_dtype)
    example_input = torch.ones(input_shape).to(param.torch_dtype)

    compressed_model = nncf.compress_weights(
        model,
        mode=param.mode,
        dataset=nncf.Dataset([example_input]),
        compression_format=nncf.CompressionFormat.FQ_LORA,
        advanced_parameters=nncf.AdvancedCompressionParameters(lora_adapter_rank=1),
        **param.extra_arguments,
    )

    check_compression_modules(compressed_model, expected_class=BaseQuantizer)
    assert compressed_model.linear.weight.dtype == param.torch_dtype

    pack_weight_spy = mocker.spy(param.decompressor_class, "pack_weight")
    with torch.no_grad():
        compressed_output = compressed_model(example_input)
        strip_compressed_model = nncf.strip(
            compressed_model, do_copy=True, strip_format=StripFormat.DQ, example_input=example_input
        )
        stripped_output = strip_compressed_model(example_input)
        assert pack_weight_spy.call_count == param.num_call_pack_weight
        assert strip_compressed_model.linear.weight.dtype == param.weight_dtype

        check_compression_modules(strip_compressed_model, param.decompressor_class)
        assert torch.allclose(compressed_output, stripped_output, atol=param.torch_atol)

        example_input = example_input.type(torch.float32)
        hook_storage = get_hook_storage(strip_compressed_model)
        for _, module in hook_storage.named_hooks():
            if isinstance(module, BaseWeightsDecompressor):
                module.result_dtype = torch.float32
        ov_model = convert_model(strip_compressed_model, example_input=example_input)
        compiled_model = _compile_ov_model(ov_model, device_name="CPU", config={inference_precision(): ov.Type.f32})
        infer_request = compiled_model.create_infer_request()
        res = infer_request.infer(example_input)
        out_name = compiled_model.outputs[0]
        ov_output = torch.from_numpy(res[out_name])
        assert torch.allclose(compressed_output.type(torch.float32), ov_output, atol=param.ov_atol)


SIGNED_WEIGHT_SAMPLE = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75]
SCALE_SAMPLE = [2.0]


@dataclass
class ParamSymFQ:
    num_bits: int
    ref_scale: list[float]
    torch_dtype: torch.dtype

    def __str__(self) -> str:
        return f"{self.num_bits}_{self.torch_dtype}"


@pytest.mark.parametrize(
    "param",
    (
        ParamSymFQ(4, [0.25], torch.float32),
        ParamSymFQ(8, [0.01563], torch.float32),
        ParamSymFQ(4, [0.25], torch.float16),
        ParamSymFQ(8, [0.01563], torch.float16),
        ParamSymFQ(4, [0.25], torch.bfloat16),
        ParamSymFQ(8, [0.01563], torch.bfloat16),
    ),
    ids=str,
)
def test_sym_fq_to_decompressor(param: ParamSymFQ):
    weights_shape = (1, len(SIGNED_WEIGHT_SAMPLE))
    weight = torch.tensor(SIGNED_WEIGHT_SAMPLE)
    weight = weight.expand(weights_shape).to(param.torch_dtype)

    scale_shape = (1, 1)
    scale = torch.tensor(SCALE_SAMPLE)
    scale = scale.expand(scale_shape)

    # reference scale calculates with this formula:
    # levels = (2 ** num_bits)
    # level_low = -(levels // 2)
    # ref_scale = SCALE_SAMPLE / abs(level_low)
    ref_scale = torch.tensor(param.ref_scale)
    ref_scale = ref_scale.expand(scale_shape).to(torch.float16)

    qspec = PTQuantizerSpec(
        num_bits=param.num_bits,
        mode=QuantizationScheme.SYMMETRIC,
        signedness_to_force=True,
        narrow_range=False,
        scale_shape=scale.shape,
        logarithm_scale=False,
        half_range=False,
        is_quantized_on_export=True,
    )
    lspec = PTLoraSpec(
        lora_rank=1,
        orig_weight_shape=weight.shape,
        weight_shape=weight.shape,
    )

    quantizer = SymmetricLoraQuantizer(qspec, lspec)
    quantizer.scale.data = scale

    with torch.no_grad():
        decompressor, q_weight = sym_fq_to_decompressor(
            quantizer,
            weight,
        )
        fq_weight = quantizer(weight)
    packed_tensor = decompressor.pack_weight(q_weight)
    qdq_weight = decompressor(packed_tensor)

    assert torch.allclose(fq_weight, qdq_weight)
    assert torch.allclose(qdq_weight, weight)
    assert torch.allclose(decompressor._scale, ref_scale)


UNSIGNED_WEIGHT_SAMPLE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
INPUT_LOW_SAMPLE = [0.0]
INPUT_RANGE_SAMPLE = [0.5]


@dataclass
class ParamAsymFQ:
    num_bits: int
    ref_scale: float
    ref_zero_point: float
    torch_dtype: torch.dtype
    atol: float

    def __str__(self) -> str:
        return f"{self.num_bits}_{self.torch_dtype}"


@pytest.mark.parametrize(
    ("param"),
    (
        ParamAsymFQ(4, 0.03333, 0.0, torch.float32, 1e-3),
        ParamAsymFQ(8, 0.00196, 0.0, torch.float32, 1e-4),
        ParamAsymFQ(4, 0.03333, 0.0, torch.float16, 1e-3),
        ParamAsymFQ(8, 0.00196, 0.0, torch.float16, 1e-8),
        ParamAsymFQ(4, 0.03333, 0.0, torch.bfloat16, 1e-8),
        ParamAsymFQ(8, 0.00196, 0.0, torch.bfloat16, 1e-8),
    ),
    ids=str,
)
def test_asym_fq_to_decompressor(param: ParamAsymFQ):
    weights_shape = (1, len(UNSIGNED_WEIGHT_SAMPLE))
    weight = torch.tensor(UNSIGNED_WEIGHT_SAMPLE)
    weight = weight.expand(weights_shape).to(param.torch_dtype)

    scale_shape = weights_shape
    # reference scale calculates with this formula:
    # levels = (2 ** num_bits)
    # level_high = levels - 1
    # ref_scale = INPUT_RANGE_SAMPLE / level_high
    ref_scale = torch.tensor(param.ref_scale)
    ref_scale = ref_scale.expand(scale_shape).to(torch.float16)

    # reference zero point calculates with this formula:
    # level_low = 0
    # ref_zero_point = level_low - round(INPUT_LOW_SAMPLE / ref_scale)
    ref_zero_point = torch.tensor(param.ref_zero_point)
    ref_zero_point = ref_zero_point.expand(scale_shape).to(torch.uint8)

    input_low = torch.tensor(INPUT_LOW_SAMPLE)
    input_low = input_low.expand(scale_shape)

    input_range = torch.tensor(INPUT_RANGE_SAMPLE)
    input_range = input_range.expand(scale_shape)

    qspec = PTQuantizerSpec(
        num_bits=param.num_bits,
        mode=QuantizationScheme.ASYMMETRIC,
        signedness_to_force=False,
        narrow_range=False,
        scale_shape=scale_shape,
        logarithm_scale=False,
        half_range=False,
        is_quantized_on_export=True,
    )
    lspec = PTLoraSpec(
        lora_rank=1,
        orig_weight_shape=weight.shape,
        weight_shape=weight.shape,
    )

    quantizer = AsymmetricLoraQuantizer(qspec, lspec)
    quantizer.input_low.data = input_low
    quantizer.input_range.data = input_range

    with torch.no_grad():
        decompressor, q_weight = asym_fq_to_decompressor(
            quantizer,
            weight,
        )
        fq_weight = quantizer(weight)
    packed_tensor = decompressor.pack_weight(q_weight)
    ref_zero_point = decompressor.pack_weight(ref_zero_point)
    qdq_weight = decompressor(packed_tensor)

    assert torch.allclose(fq_weight, qdq_weight, atol=param.atol)
    assert torch.allclose(qdq_weight, weight, atol=param.atol)
    assert torch.allclose(decompressor._zero_point, ref_zero_point)
    assert torch.allclose(decompressor._scale, ref_scale)
