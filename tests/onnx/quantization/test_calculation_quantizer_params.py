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

import numpy as np
import pytest

from nncf.onnx.quantization.quantizer_parameters import get_level_low_level_high
from nncf.quantization.fake_quantize import calculate_scale_zero_point
from nncf.tensor import Tensor
from tests.cross_fw.test_templates.test_calculate_quantizer_parameters import TemplateTestFQParams

EPS = np.finfo(np.float32).eps


@pytest.mark.parametrize(
    ("inp_low, inp_high, level_low, level_high, narrow_range, ref_scale, ref_zero_point"),
    (
        (-8e-05, 8e-05, -128, 127, True, 6.3e-7, 0),
        (-0.0008062992, 0.0008, 0, 255, False, 6.3e-6, 128),
        (-10.019569, 10, 0, 1023, False, 0.01956947, 512),
        (0, 1, 0, 255, False, 0.00392157, 0),
        (-1, 10, 0, 1023, False, 0.01075269, 93),
        (-11.428572, 10, 0, 15, False, 1.4285715, 8),
        (-10, 10, -512, 511, False, 0.01955034, 0),
        (-10, 10, -128, 127, True, 0.07874016, 0),
        (0, 25, 0, 15, False, 1.6666666, 0),
        (1, 1, -128, 127, True, EPS, -128),
        (0, 0, -128, 127, True, EPS, -127),
        (np.array([0, 1]), np.array([0, 1]), -128, 127, True, np.array([EPS, EPS]), np.array([-127, -128])),
    ),
)
def test_calculate_scale_zero_point(inp_low, inp_high, level_low, level_high, narrow_range, ref_scale, ref_zero_point):
    inp_low, inp_high = Tensor(np.array(inp_low)), Tensor(np.array(inp_high))
    scale, zero_point = calculate_scale_zero_point(inp_low, inp_high, level_low, level_high, narrow_range)
    ref_zero_point = np.array(ref_zero_point, dtype=np.int32)
    assert np.allclose(ref_scale, scale.data)
    assert np.allclose(ref_zero_point, zero_point.data)


@pytest.mark.parametrize("num_bits, tensor_type, ref_levels", ((8, np.int8, (-128, 127)), (8, np.uint8, (0, 255))))
def test_calculate_levels(num_bits, tensor_type, ref_levels):
    assert (ref_levels[0], ref_levels[1]) == get_level_low_level_high(tensor_type)


class TestFQParams(TemplateTestFQParams):
    def to_nncf_tensor(self, t):
        return Tensor(t)
