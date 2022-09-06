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

from nncf.experimental.onnx.common import infer_input_shape
from tests.common.helpers import TEST_ROOT
from tests.onnx.quantization.common import ModelToTest


TEST_CASES = [
    (ModelToTest('densenet-7', [1, 3, 224, 224]), 'data_0'),
    (ModelToTest('densenet-8', [1, 3, 224, 224]), 'data_0'),
    (ModelToTest('densenet-9', [1, 3, 224, 224]), 'data_0'),
]
DET_TEST_CASES = [
    (ModelToTest('MaskRCNN-12', [3, 30, 30]), 'image'),
    (ModelToTest('yolov3-12', [1, 3, 30, 30]), 'input_1'),
]


def load_model(model_to_test):
    onnx_model_dir = str(TEST_ROOT.joinpath('onnx', 'data', 'models'))
    onnx_model_path = str(TEST_ROOT.joinpath(onnx_model_dir, model_to_test.model_name + '.onnx'))
    if not os.path.isdir(onnx_model_dir):
        os.mkdir(onnx_model_dir)

    original_model = onnx.load(onnx_model_path)

    return original_model


class TestMultipleInputs:

    @pytest.mark.parametrize(('model_to_test', 'input_name'), TEST_CASES)
    def test_input_shape_input_name(self, model_to_test, input_name):
        model = load_model(model_to_test)

        result_input_shape, result_input_name = infer_input_shape(model, model_to_test.input_shape, input_name)

        assert isinstance(result_input_shape, (list, tuple)) and result_input_name is not None
        assert result_input_shape == model_to_test.input_shape
        assert result_input_name == input_name

    @pytest.mark.parametrize(('model_to_test', 'input_name'), TEST_CASES)
    def test_input_shape(self, model_to_test, input_name):
        model = load_model(model_to_test)

        result_input_shape, result_input_name = infer_input_shape(model, model_to_test.input_shape, None)

        assert isinstance(result_input_shape, (list, tuple)) and result_input_name is not None
        assert result_input_name == input_name

    @pytest.mark.parametrize(('model_to_test', 'input_name'), TEST_CASES)
    def test_input_name(self, model_to_test, input_name):
        model = load_model(model_to_test)

        result_input_shape, result_input_name = infer_input_shape(model, None, input_name)
        if result_input_shape is None:
            assert pytest.skip(reason='For Models with dynamic input_shape, input_shape must be set.')

        assert isinstance(result_input_shape, (list, tuple)) and result_input_name is not None
        assert result_input_shape == model_to_test.input_shape

    @pytest.mark.xfail(reason="both input_shape and input_name are None.")
    @pytest.mark.parametrize(('model_to_test', 'input_name'), TEST_CASES)
    def test_input_shape_input_name_none(self, model_to_test, input_name):
        model = load_model(model_to_test)

        _, _ = infer_input_shape(model, None, None)


class TestDynamicInputShape:

    @pytest.mark.parametrize(('model_to_test', 'input_name'), DET_TEST_CASES)
    def test_input_shape_input_name(self, model_to_test, input_name):
        model = load_model(model_to_test)

        result_input_shape, result_input_name = infer_input_shape(model, model_to_test.input_shape, input_name)

        assert isinstance(result_input_shape, (list, tuple)) and result_input_name is not None
        assert result_input_shape == model_to_test.input_shape
        assert result_input_name == input_name

    @pytest.mark.xfail(reason="For models with dynamic input_shape, input_shape and input_name must be set.")
    @pytest.mark.parametrize(('model_to_test', 'input_name'), DET_TEST_CASES)
    def test_input_shape(self, model_to_test, input_name):
        model = load_model(model_to_test)

        _, _ = infer_input_shape(model, model_to_test.input_shape, None)

    @pytest.mark.xfail(reason="For models with dynamic input_shape, input_shape and input_name must be set.")
    @pytest.mark.parametrize(('model_to_test', 'input_name'), DET_TEST_CASES)
    def test_input_name(self, model_to_test, input_name):
        model = load_model(model_to_test)

        _, _ = infer_input_shape(model, None, input_name)

    @pytest.mark.xfail(reason="For models with dynamic input_shape, input_shape and input_name must be set.")
    @pytest.mark.parametrize(('model_to_test', 'input_name'), DET_TEST_CASES)
    def test_input_shape_input_name_none(self, model_to_test, input_name):
        model = load_model(model_to_test)

        _, _ = infer_input_shape(model, None, None)
