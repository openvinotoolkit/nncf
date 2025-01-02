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

import pytest
import torch

from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.torch.quantization.layers import QUANTIZATION_MODULES
from nncf.torch.quantization.layers import PTQuantizerSpec


@pytest.mark.parametrize("quantizer_cls", QUANTIZATION_MODULES.values())
def test_quantizer_layers_accepts_return_type(quantizer_cls):
    actual_input = torch.range(0, 10)
    input_ = torch.return_types.max((actual_input, actual_input))
    quantizer_spec = PTQuantizerSpec(
        num_bits=8,
        mode=QuantizationMode.SYMMETRIC,
        signedness_to_force=True,
        narrow_range=True,
        half_range=True,
        scale_shape=(1,),
        logarithm_scale=False,
    )
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
