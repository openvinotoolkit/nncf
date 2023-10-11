# Copyright (c) 2023 Intel Corporation
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
from typing import Tuple

import numpy as np
import onnx
import pytest

from nncf.onnx.graph.onnx_helper import get_tensor_value
from nncf.onnx.quantization.quantize_model import compress_quantize_weights_transformation
from tests.onnx.models import LinearModel
from tests.onnx.quantization.common import ptq_quantize_model


@dataclass
class TestCase:
    model: onnx.ModelProto
    ref_weight_names: Tuple[str]


@pytest.mark.parametrize("test_case", [TestCase(model=LinearModel(), ref_weight_names=("Conv1_W", "Conv2_W"))])
def test_model_has_int8_weights(test_case):
    onnx_model = test_case.model.onnx_model
    quantized_model = ptq_quantize_model(onnx_model)
    quantized_model_with_int8_weights = compress_quantize_weights_transformation(quantized_model)
    for weight_name in test_case.ref_weight_names:
        tensor = get_tensor_value(quantized_model_with_int8_weights, weight_name)
        assert tensor.dtype == np.int8


# TODO: add test on weights values
