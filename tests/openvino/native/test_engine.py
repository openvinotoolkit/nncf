# Copyright (c) 2025 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import wraps

import numpy as np
import pytest

import nncf
from nncf.openvino.engine import OVNativeEngine
from tests.openvino.native.models import ConvModel
from tests.openvino.native.models import LinearModel
from tests.openvino.native.models import QuantizedModel
from tests.openvino.native.models import StatefulModel


def check_engine_creation_and_inference(model, input_data):
    engine = OVNativeEngine(model)
    outputs = engine.infer(input_data)
    for result in model.get_results():
        res_name = result.get_output_tensor(0).get_any_name()
        assert res_name in outputs
        assert outputs[res_name].shape == tuple(result.get_output_shape(0))


@pytest.mark.parametrize("model_creator_func", [LinearModel, ConvModel])
def test_infer_original_model_dict(model_creator_func):
    model = model_creator_func().ov_model
    input_data = {inp.get_any_name(): np.random.rand(*inp.shape) for inp in model.inputs}
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


@pytest.mark.parametrize("stateful", [True, False])
def test_compiled_model_engine_inference_stateful(stateful):
    model = StatefulModel(stateful).ov_model
    input_data = [np.ones(inp.shape) for inp in model.get_parameters()]

    engine = OVNativeEngine(model)

    for _ in range(10):
        engine.infer(input_data)

    out = engine.infer(input_data)

    input_data = input_data[0]
    out = out["Result"]

    assert np.array_equal(out[0], input_data[0])


def test_stateful_model_inference_with_controlled_resetting():
    def wrap_reset_state(infer_request):
        nonlocal reset_order
        original_reset_state = infer_request.reset_state

        @wraps(infer_request.reset_state)
        def _reset_state():
            reset_order.append("reset")
            original_reset_state()

        infer_request.reset_state = _reset_state

    model = StatefulModel(True).ov_model
    inp = model.get_parameters()[0]
    input_data = [{"input_data": np.ones(inp.shape), nncf.Dataset.RESET_STATE_KEY: False} for _ in range(10)]
    reset_ind = [2, 5, 7]
    for ind in reset_ind:
        input_data[ind][nncf.Dataset.RESET_STATE_KEY] = True

    engine = OVNativeEngine(model)
    reset_order = []
    wrap_reset_state(engine.engine.infer_request)

    for inp_data in input_data:
        engine.infer(inp_data)
        reset_order.append("infer")

    assert reset_order == [
        "infer",
        "infer",
        "reset",
        "infer",
        "infer",
        "infer",
        "reset",
        "infer",
        "infer",
        "reset",
        "infer",
        "infer",
        "infer",
    ]
