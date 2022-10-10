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

import torch
from torchvision import models
import onnx
# pylint: disable=no-member

from tests.common.helpers import TEST_ROOT
from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import ptq_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph_onnx_models


@pytest.mark.parametrize(('model_to_test', 'model'),
                         [(ModelToTest('resnet18', [1, 3, 224, 224]), models.resnet18()),
                          (ModelToTest('mobilenet_v2', [1, 3, 224, 224]), models.mobilenet_v2()),
                          ]
                         )
def test_min_max_ptq_quantization_graph_are_same(tmp_path, model_to_test, model):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(model_to_test.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    min_max_quantized_model = min_max_quantize_model(model_to_test.input_shape, original_model)
    ptq_quantized_model = ptq_quantize_model(model_to_test.input_shape, original_model)
    compare_nncf_graph_onnx_models(min_max_quantized_model, ptq_quantized_model)
