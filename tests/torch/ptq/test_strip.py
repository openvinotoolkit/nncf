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
from torch.quantization import FakeQuantize

import nncf
from tests.torch.helpers import BasicConvTestModel


@pytest.mark.parametrize("strip_type", ("nncf", "torch", "nncf_interfere"))
@pytest.mark.parametrize("do_copy", (True, False), ids=["copy", "inplace"])
def test_nncf_strip_api(strip_type, do_copy):
    model = BasicConvTestModel()
    quantized_model = nncf.quantize(model, nncf.Dataset([torch.ones(model.INPUT_SIZE)]), subset_size=1)

    if strip_type == "nncf":
        strip_model = nncf.strip(quantized_model, do_copy)
    elif strip_type == "torch":
        strip_model = nncf.torch.strip(quantized_model, do_copy)
    elif strip_type == "nncf_interfere":
        strip_model = quantized_model.nncf.strip(do_copy)

    if do_copy:
        assert id(strip_model) != id(quantized_model)
    else:
        assert id(strip_model) == id(quantized_model)

    for fq in strip_model.nncf.external_quantizers.values():
        assert isinstance(fq, FakeQuantize)
