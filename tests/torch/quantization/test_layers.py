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

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import PTLoraNLSSpec
from nncf.torch.quantization.layers import PTLoraSpec
from nncf.torch.quantization.layers import PTQuantizerSpec


@pytest.mark.parametrize("registred", list(QUANTIZATION_MODULES.registry_dict.items()))
def test_quantizer_layers_accepts_return_type(registred):
    mode, quantizer_cls = registred

    actual_input = torch.range(0, 10)
    input_ = torch.return_types.max((actual_input, actual_input))

    quantizer_spec = PTQuantizerSpec(
        num_bits=8,
        mode=mode,
        signedness_to_force=True,
        narrow_range=True,
        half_range=True,
        scale_shape=(1,),
        logarithm_scale=False,
    )
    if mode not in [QuantizationMode.ASYMMETRIC, QuantizationMode.SYMMETRIC]:
        shape = actual_input.unsqueeze(dim=0).shape
        lora_spec = PTLoraSpec(0, shape, shape)
        if mode in [QuantizationMode.ASYMMETRIC_LORA_NLS, QuantizationMode.SYMMETRIC_LORA_NLS]:
            lora_spec = PTLoraNLSSpec(0, 0, shape, shape)
        quantizer = quantizer_cls(quantizer_spec, lora_spec)
    else:
        quantizer = quantizer_cls(quantizer_spec)

    visited = False

    def check_types(fn):
        def wrapped(x: torch.Tensor):
            assert isinstance(x, torch.Tensor)
            nonlocal visited
            visited = True
            return fn(x)

        return wrapped

    quantizer._forward_impl = check_types(quantizer._forward_impl)
    quantizer(input_)
    assert visited
