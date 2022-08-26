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

# pylint: disable=redefined-outer-name

import pytest

import os

import torch
from torchvision import models
import onnx

from nncf.experimental.onnx.datasets.common import infer_input_shape
from tests.common.helpers import TEST_ROOT
from tests.onnx.quantization.common import ModelToTest


MODEL_NAMES = [
    'resnet18',
    'mobilenet_v2',
    'mobilenet_v3_small',
    'inception_v3',
    'googlenet',
    'vgg16',
    'shufflenet_v2_x1_0',
    'squeezenet1_0',
    'densenet121',
    'mnasnet0_5',
]

MODELS = [
    models.resnet18(),
    models.mobilenet_v2(),
    models.mobilenet_v3_small(),
    models.inception_v3(),
    models.googlenet(),
    models.vgg16(),
    models.shufflenet_v2_x1_0(),
    models.squeezenet1_0(),
    models.densenet121(),
    models.mnasnet0_5(),
]

INPUT_SHAPE = [1, 3, 224, 224]
INPUT_KEY = 'input.1'
LOW_OPSET_VERSIONS = [7, 8, 9]


TEST_CASES = []
for name, model in zip(MODEL_NAMES, MODELS):
    for version in LOW_OPSET_VERSIONS:
        TEST_CASES.append((ModelToTest(name, INPUT_SHAPE), model, INPUT_KEY, version))


def load_model(model_to_test, model, opset_version):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)
    x = torch.randn(model_to_test.input_shape, requires_grad=False)
    torch.onnx.export(model, x, onnx_model_path, opset_version=opset_version)

    original_model = onnx.load(onnx_model_path)

    return original_model


class TestLowOpsetModel:

    @pytest.mark.parametrize(('model_to_test', 'model', 'input_keys', 'opset_version'), TEST_CASES)
    def test_input_shape(self, model_to_test, model, input_keys, opset_version):
        model = load_model(model_to_test, model, opset_version)

        input_shape, input_keys = infer_input_shape(model, model_to_test.input_shape, None)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.parametrize(('model_to_test', 'model', 'input_keys', 'opset_version'), TEST_CASES)
    def test_input_keys(self, model_to_test, model, input_keys, opset_version):
        model = load_model(model_to_test, model, opset_version)

        input_shape, input_keys = infer_input_shape(model, None, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.parametrize(('model_to_test', 'model', 'input_keys', 'opset_version'), TEST_CASES)
    def test_input_shape_input_keys(self, model_to_test, model, input_keys, opset_version):
        model = load_model(model_to_test, model, opset_version)

        input_shape, input_keys = infer_input_shape(model, model_to_test.input_shape, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.xfail(reason="both input_shape and input_keys are None")
    @pytest.mark.parametrize(('model_to_test', 'model', 'input_keys', 'opset_version'), TEST_CASES)
    def test_input_shape_input_keys_none(self, model_to_test, model, input_keys, opset_version):
        model = load_model(model_to_test, model, opset_version)

        _, _ = infer_input_shape(model, None, None)
