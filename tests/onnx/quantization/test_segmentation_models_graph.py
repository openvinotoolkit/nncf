"""
 Copyright (c) 2023 Intel Corporation
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

import pytest

from tests.onnx.conftest import ONNX_MODEL_DIR
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.weightless_model import load_model_topology_with_zeros_weights
from tests.onnx.quantization.common import mock_collect_statistics


@pytest.mark.parametrize(('model_to_test'),
                         [ModelToTest('icnet_camvid', [1, 3, 768, 960]),
                          ModelToTest('unet_camvid', [1, 3, 368, 480])]
                         )
def test_min_max_quantization_graph(tmp_path, mocker, model_to_test):
    mock_collect_statistics(mocker)
    if model_to_test.model_name == 'unet_camvid':
        pytest.skip('Ticket 96049')
    onnx_model_path = ONNX_MODEL_DIR / (model_to_test.model_name + '.onnx')

    original_model = load_model_topology_with_zeros_weights(onnx_model_path)
    quantized_model = min_max_quantize_model(model_to_test.input_shape, original_model)
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)
