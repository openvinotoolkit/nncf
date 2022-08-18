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
from tests.onnx.quantization.common import TestCase
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import infer_model


@pytest.mark.parametrize(('test_case', 'model'),
                         [(TestCase('resnet18', [1, 3, 224, 224]), models.resnet18()),
                          (TestCase('mobilenet_v2', [1, 3, 224, 224]), models.mobilenet_v2()),
                          (TestCase('mobilenet_v3_small', [1, 3, 224, 224]), models.mobilenet_v3_small()),
                          (TestCase('inception_v3', [1, 3, 224, 224]), models.inception_v3()),
                          (TestCase('googlenet', [1, 3, 224, 224]), models.googlenet()),
                          (TestCase('vgg16', [1, 3, 224, 224]), models.vgg16()),
                          (TestCase('shufflenet_v2_x1_0', [1, 3, 224, 224]), models.shufflenet_v2_x1_0()),
                          (TestCase('squeezenet1_0', [1, 3, 224, 224]), models.squeezenet1_0()),
                          (TestCase('densenet121', [1, 3, 224, 224]), models.densenet121()),
                          (TestCase('mnasnet0_5', [1, 3, 224, 224]), models.mnasnet0_5()),
                          ]
                         )
def test_min_max_quantization_graph(tmp_path, test_case, model):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, test_case.model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(test_case.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    quantized_model = min_max_quantize_model(test_case.input_shape, original_model)
    compare_nncf_graph(quantized_model, test_case.path_ref_graph)
    infer_model(test_case.input_shape, quantized_model)
