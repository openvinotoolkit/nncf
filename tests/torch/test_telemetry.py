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

from nncf import NNCFConfig
from tests.cross_fw.shared.helpers import telemetry_send_event_test_driver
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test


def test_telemetry_is_sent(mocker):
    def use_nncf_fn():
        config = NNCFConfig({"input_info": {"sample_size": [1, 1, 32, 32]}})
        create_compressed_model_and_algo_for_test(TwoConvTestModel(), config)

    telemetry_send_event_test_driver(mocker, use_nncf_fn)
