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

import numpy as np
import pytest
from openvino.runtime import opset9 as opset

from nncf.openvino.graph.model_utils import create_bias_constant_value


def get_conv_node(input_shape, dtype):
    input_node = opset.parameter(input_shape, dtype=dtype)
    strides = [1, 1]
    pads = [0, 0]
    dilations = [1, 1]
    return opset.convolution(
        input_node, np.zeros((4, input_shape[1], 1, 1), dtype=dtype), strides, pads, pads, dilations
    )


@pytest.mark.parametrize(
    "input_shape,dtype",
    [((2, 3, 4, 5), np.float32), ((1, 1, 1, 1), np.float64)],
)
def test_create_bias_constant_value(input_shape, dtype):
    conv = get_conv_node(input_shape, dtype)
    val = create_bias_constant_value(conv, 5)
    assert val.shape == (1, 4, 1, 1)
    assert np.equal(val, np.full((1, 4, 1, 1), 5)).all()
