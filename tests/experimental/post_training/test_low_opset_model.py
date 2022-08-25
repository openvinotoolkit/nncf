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

from nncf.experimental.onnx.datasets.imagenet_dataset import infer_input_shape
from tests.common.helpers import TEST_ROOT


MODEL_NAMES = [
    'resnet18',
    'mobilenet_v2',
    'inception_v3',
    'googlenet',
    'vgg16',
    'densenet121'
]

MODELS = [
    models.resnet18(),
    models.mobilenet_v2(),
    models.inception_v3(),
    models.googlenet(),
    models.vgg16(),
    models.densenet121(),
]

INPUT_SHAPES = [
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
    [1, 3, 224, 224],
]

INPUT_KEYS = [
    "input.1",
    "input.1",
    "input.1",
    "input.1",
    "input.1",
    "input.1",
]

TEST_CASES = [
    pytest.param(name, model, shape, key)
    for name, model, shape, key in zip(MODEL_NAMES, MODELS, INPUT_SHAPES, INPUT_KEYS)
]


@pytest.fixture
def load_low_opset_model():
    def _load_low_opset_model(model_name, model, input_shape):
        onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
        onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_name + '.onnx'))
        if not os.path.isdir(onnx_model_dir):
            os.mkdir(onnx_model_dir)
        x = torch.randn(input_shape, requires_grad=False)
        torch.onnx.export(model, x, onnx_model_path, opset_version=8)

        original_model = onnx.load(onnx_model_path)

        return original_model

    return _load_low_opset_model


class TestLowOpsetModel:

    @pytest.mark.parametrize(('model_name', 'model', 'input_shape', 'input_keys'), TEST_CASES)
    def test_input_shape(self, load_low_opset_model, model_name, model, input_shape, input_keys):
        model = load_low_opset_model(model_name, model, input_shape)

        input_keys = None

        input_shape, input_keys = infer_input_shape(model, input_shape, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.parametrize(('model_name', 'model', 'input_shape', 'input_keys'), TEST_CASES)
    def test_input_keys(self, load_low_opset_model, model_name, model, input_shape, input_keys):
        model = load_low_opset_model(model_name, model, input_shape)

        input_shape = None

        input_shape, input_keys = infer_input_shape(model, input_shape, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.parametrize(('model_name', 'model', 'input_shape', 'input_keys'), TEST_CASES)
    def test_input_shape_input_keys(self, load_low_opset_model, model_name, model, input_shape, input_keys):
        model = load_low_opset_model(model_name, model, input_shape)

        input_shape, input_keys = infer_input_shape(model, input_shape, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.xfail(reason="both input_shape and input_keys are None")
    @pytest.mark.parametrize(('model_name', 'model', 'input_shape', 'input_keys'), TEST_CASES)
    def test_input_shape_input_keys_none(self, load_low_opset_model, model_name, model, input_shape, input_keys):
        model = load_low_opset_model(model_name, model, input_shape)

        input_shape = None
        input_keys = None

        input_shape, input_keys = infer_input_shape(model, input_shape, input_keys)
