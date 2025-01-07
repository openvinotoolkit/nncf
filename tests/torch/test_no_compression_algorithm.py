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
from tests.torch.helpers import PTTensorListComparator
from tests.torch.helpers import TwoConvTestModel
from tests.torch.helpers import create_compressed_model_and_algo_for_test

EPS = 1e-9
INPUT_SIZE = [1, 4, 4]

NO_COMPRESSION_NNCF_CONFIG = NNCFConfig({"model": "basic_config", "input_info": {"sample_size": [1] + INPUT_SIZE}})


def test_no_compression_algo_not_change_model_params():
    orig_model = TwoConvTestModel()
    model, _algo = create_compressed_model_and_algo_for_test(orig_model, NO_COMPRESSION_NNCF_CONFIG)

    orig_model_state = orig_model.state_dict()
    model_state = model.state_dict()
    PTTensorListComparator.check_equal(list(orig_model_state.values()), list(model_state.values()))
