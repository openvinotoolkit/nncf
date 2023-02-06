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
import numpy as np

from nncf.experimental.openvino_native.engine import OVNativeEngine

from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import QuantizedModel


def check_engine_creation_and_inference(model, input_data):
    engine = OVNativeEngine(model)
    outputs = engine.infer(input_data)
    for result in model.get_results():
        res_name = result.get_friendly_name()
        assert res_name in outputs.keys()
        assert outputs[res_name].shape == tuple(result.get_output_shape(0))


@pytest.mark.parametrize("model_creator_func", [LinearModel, ConvModel])
def test_infer_original_model_dict(model_creator_func):
    model = model_creator_func().ov_model
    input_data = {inp.get_friendly_name(): np.random.rand(*inp.shape) for inp in model.get_parameters()}
    check_engine_creation_and_inference(model, input_data)


@pytest.mark.parametrize("model_creator_func", [LinearModel, ConvModel])
def test_infer_original_model_list(model_creator_func):
    model = model_creator_func().ov_model
    input_data = [np.random.rand(*inp.shape) for inp in model.get_parameters()]
    check_engine_creation_and_inference(model, input_data)


@pytest.mark.parametrize("model_creator_func", [LinearModel, ConvModel])
def test_infer_original_model_tuple(model_creator_func):
    model = model_creator_func().ov_model
    input_data = tuple(np.random.rand(*inp.shape) for inp in model.get_parameters())
    check_engine_creation_and_inference(model, input_data)


def test_infer_original_model_array():
    model = LinearModel().ov_model
    input_data = np.random.rand(*model.get_parameters()[0].shape)
    check_engine_creation_and_inference(model, input_data)


def test_infer_quantized_model_list():
    model = QuantizedModel().ov_model
    input_data = [np.random.rand(*inp.shape) for inp in model.get_parameters()]
    check_engine_creation_and_inference(model, input_data)
