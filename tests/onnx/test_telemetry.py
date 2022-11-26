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

from tests.onnx.quantization.common import min_max_quantize_model
from tests.onnx.models import LinearModel
from tests.shared.helpers import telemetry_send_event_test_driver

def test_telemetry_is_sent(mocker):
    def use_nncf_fn():
        model_to_test = LinearModel()
        _ = min_max_quantize_model(model_to_test.input_shape[0], model_to_test.onnx_model)
    telemetry_send_event_test_driver(mocker, use_nncf_fn)
