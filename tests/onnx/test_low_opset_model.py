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

import onnx

from nncf.experimental.onnx.datasets.common import infer_input_shape
from tests.common.helpers import TEST_ROOT
from tests.onnx.quantization.common import ModelToTest


MODEL_NAMES = [
    'densenet-7',
    'densenet-8',
    'densenet-9',
]

INPUT_SHAPE = [1, 3, 224, 224]
INPUT_KEY = 'data_0'

TEST_CASES = [(ModelToTest(name, INPUT_SHAPE), INPUT_KEY) for name in MODEL_NAMES]


def load_model(model_to_test):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)

    original_model = onnx.load(onnx_model_path)

    return original_model


class TestLowOpsetModel:

    @pytest.mark.parametrize(('model_to_test', 'input_keys'), TEST_CASES)
    def test_input_shape(self, model_to_test, input_keys):
        model = load_model(model_to_test)

        input_shape, input_keys = infer_input_shape(model, model_to_test.input_shape, None)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.parametrize(('model_to_test', 'input_keys'), TEST_CASES)
    def test_input_keys(self, model_to_test, input_keys):
        model = load_model(model_to_test)

        input_shape, input_keys = infer_input_shape(model, None, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.parametrize(('model_to_test', 'input_keys'), TEST_CASES)
    def test_input_shape_input_keys(self, model_to_test, input_keys):
        model = load_model(model_to_test)

        input_shape, input_keys = infer_input_shape(model, model_to_test.input_shape, input_keys)

        assert isinstance(input_shape, (list, tuple)) and len(input_shape) == 4
        assert isinstance(input_keys, list)

    @pytest.mark.xfail(reason="both input_shape and input_keys are None")
    @pytest.mark.parametrize(('model_to_test', 'input_keys'), TEST_CASES)
    def test_input_shape_input_keys_none(self, model_to_test, input_keys):
        model = load_model(model_to_test)

        _, _ = infer_input_shape(model, None, None)
