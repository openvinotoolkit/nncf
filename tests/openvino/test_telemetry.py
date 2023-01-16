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
from tests.shared.datasets import MockDataset
from tests.shared.helpers import telemetry_send_event_test_driver

from openvino.runtime import Model, Shape, Type, op, opset8

import nncf
from nncf import Dataset

INPUT_SHAPE = [2, 1, 1, 1]
def get_mock_model() -> Model:
    param_node = op.Parameter(Type.f32, Shape(INPUT_SHAPE))
    softmax_axis = 1
    softmax_node = opset8.softmax(param_node, softmax_axis)
    return Model(softmax_node, [param_node], 'mock')


def test_telemetry_is_sent(mocker):
    def use_nncf_fn():
        model_to_test = get_mock_model()

        _ = nncf.quantize(model_to_test, Dataset(MockDataset(INPUT_SHAPE)))
    telemetry_send_event_test_driver(mocker, use_nncf_fn)
