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

import pytest

import os

import onnx

from tests.common.helpers import TEST_ROOT
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import infer_model


@pytest.mark.parametrize(('model_to_test'),
                         [ModelToTest('icnet_camvid', [1, 3, 768, 960]),
                          ModelToTest('unet_camvid', [1, 3, 368, 480]),
                          ]
                         )
def test_min_max_quantization_graph(tmp_path, model_to_test):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)

    original_model = onnx.load(onnx_model_path)
    quantized_model = min_max_quantize_model(model_to_test.input_shape, original_model)
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)
    infer_model(model_to_test.input_shape, quantized_model)
