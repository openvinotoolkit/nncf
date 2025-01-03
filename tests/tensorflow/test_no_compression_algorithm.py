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
from tests.tensorflow.helpers import TFTensorListComparator
from tests.tensorflow.helpers import create_compressed_model_and_algo_for_test
from tests.tensorflow.helpers import get_basic_two_conv_test_model

EPS = 1e-9
INPUT_SIZE = [4, 4, 1]

NO_COMPRESSION_NNCF_CONFIG = NNCFConfig({"model": "basic_config", "input_info": {"sample_size": [1] + INPUT_SIZE}})


def test_no_compression_algo_not_change_model_params():
    orig_model = get_basic_two_conv_test_model()
    model, _algo = create_compressed_model_and_algo_for_test(orig_model, NO_COMPRESSION_NNCF_CONFIG)

    orig_model_weights = orig_model.get_weights()
    model_weights = model.get_weights()
    TFTensorListComparator.check_equal(orig_model_weights, model_weights)
