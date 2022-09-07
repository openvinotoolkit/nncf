"""
 Copyright (c) 2022 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

# pylint: disable=no-member, redefined-outer-name, no-name-in-module

import pytest

from tests.onnx.quantization.common import infer_model, min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.models import ALL_MODELS
from tests.onnx.models import MultiInputOutputModel


@pytest.mark.parametrize('model_to_test', ALL_MODELS)
def test_syntetic_models_graph(model_to_test):
    if isinstance(model_to_test, MultiInputOutputModel):
        pytest.skip('min_max_quantize_model does not support many inputs for now.')
    quantized_model = min_max_quantize_model(model_to_test.input_shape[0], model_to_test.onnx_model)
    infer_model(model_to_test.input_shape[0], quantized_model)
    compare_nncf_graph(quantized_model, 'synthetic/' + model_to_test.path_ref_graph)
