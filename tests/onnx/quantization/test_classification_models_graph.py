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

import torch
from torchvision import models
import onnx
# pylint: disable=no-member

from tests.onnx.quantization.common import ModelToTest
from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.quantization.common import compare_nncf_graph
from tests.onnx.quantization.common import mock_collect_statistics


@pytest.mark.parametrize(('model_to_test', 'model'),
                         [(ModelToTest('resnet18', [1, 3, 224, 224]), models.resnet18(pretrained=True)),
                          (ModelToTest('mobilenet_v2', [1, 3, 224, 224]), models.mobilenet_v2(pretrained=True)),
                          (ModelToTest('mobilenet_v3_small', [1, 3, 224, 224]),
                           models.mobilenet_v3_small(pretrained=True)),
                          (ModelToTest('inception_v3', [1, 3, 224, 224]), models.inception_v3(pretrained=True)),
                          (ModelToTest('googlenet', [1, 3, 224, 224]), models.googlenet(pretrained=True)),
                          (ModelToTest('vgg16', [1, 3, 224, 224]), models.vgg16(pretrained=True)),
                          (ModelToTest('shufflenet_v2_x1_0', [1, 3, 224, 224]),
                           models.shufflenet_v2_x1_0(pretrained=True)),
                          (ModelToTest('squeezenet1_0', [1, 3, 224, 224]), models.squeezenet1_0(pretrained=True)),
                          (ModelToTest('densenet121', [1, 3, 224, 224]), models.densenet121(pretrained=True)),
                          (ModelToTest('mnasnet0_5', [1, 3, 224, 224]), models.mnasnet0_5(pretrained=True)),
                          ]
                         )
def test_min_max_quantization_graph(tmp_path, mocker, model_to_test, model):
    mock_collect_statistics(mocker)
    onnx_model_path = tmp_path / model_to_test.model_name
    x = torch.randn(model_to_test.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)

    original_model = onnx.load(onnx_model_path)
    quantized_model = min_max_quantize_model(model_to_test.input_shape, original_model)
    compare_nncf_graph(quantized_model, model_to_test.path_ref_graph)
