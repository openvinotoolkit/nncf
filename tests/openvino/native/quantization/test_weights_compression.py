# Copyright (c) 2023 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import numpy as np
import openvino.runtime as ov
from nncf.quantization import compress_weights
from tests.openvino.native.models import IntegerModel
from tests.openvino.native.models import WeightsModel

TEST_MODELS = {
    IntegerModel: ["gather_2_data", "matmul_1_data", "matmul_2_data"],
    WeightsModel: ["weights_0", "weights_1"]
}


@pytest.mark.parametrize("model_creator_func", TEST_MODELS)
def test_compress_weights(model_creator_func):
    ref_compressed_weights = TEST_MODELS[model_creator_func]
    model = model_creator_func().ov_model
    compressed_model = compress_weights(model)

    n_compressed_weights = 0
    for op in compressed_model.get_ops():
        if op.get_type_name() == 'Constant' and op.get_friendly_name() in ref_compressed_weights:
            assert op.get_element_type() == ov.Type(np.int8)
            n_compressed_weights += 1

    assert n_compressed_weights == len(ref_compressed_weights)
