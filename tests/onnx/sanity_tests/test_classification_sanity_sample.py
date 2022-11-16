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

from unittest.mock import patch

import numpy as np
import onnx
import pytest
import torch
from torchvision import models

from examples.experimental.onnx.classification.onnx_ptq_classification import run
from tests.onnx.quantization.common import get_dataset_for_test
from tests.onnx.conftest import ONNX_MODEL_DIR
from tests.onnx.quantization.common import infer_model

MODEL_NAMES = [
    'resnet18',
    'mobilenet_v2',
    'inception_v3',
    'googlenet',
    'vgg16',
    'shufflenet_v2_x1_0'
]

MODELS = [
    models.resnet18(),
    models.mobilenet_v2(),
    models.inception_v3(),
    models.googlenet(),
    models.vgg16(),
    models.shufflenet_v2_x1_0(),
]

INPUT_SHAPES = [
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
]

TEST_CASES = [
    pytest.param(name, model, shape) for name, model, shape in zip(MODEL_NAMES, MODELS, INPUT_SHAPES)
]


def mock_dataloader_creator(dataset_path, input_shape, batch_size, shuffle):
    return [(np.zeros(input_shape[1:], dtype=np.float32), 0)]


def mock_dataset_creator(dataloader, input_name):
    return get_dataset_for_test(dataloader, input_name)


@pytest.mark.parametrize(("model_name, model, input_shape"), TEST_CASES)
@patch('examples.experimental.onnx.classification.onnx_ptq_classification.create_dataloader',
       new=mock_dataloader_creator)
@patch('examples.experimental.onnx.classification.onnx_ptq_classification.create_dataset',
       new=mock_dataset_creator)
def test_sanity_quantize_sample(tmp_path, model_name, model, input_shape):
    onnx_model_path = ONNX_MODEL_DIR / (model_name + '.onnx')
    x = torch.randn(input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=13)
    onnx_output_model_path = str(tmp_path / model_name)
    run(str(onnx_model_path), onnx_output_model_path, 'none',
        batch_size=1, shuffle=True, num_init_samples=1,
        input_shape=input_shape, ignored_scopes=None)
    quantized_model = onnx.load_model(onnx_output_model_path)
    infer_model(input_shape, quantized_model)
