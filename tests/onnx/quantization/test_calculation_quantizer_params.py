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
from nncf.common.quantization.structs import QuantizationMode
from nncf.onnx.quantization.quantizer_parameters import calculate_scale_zero_point
from nncf.onnx.quantization.quantizer_parameters import get_level_low_level_high


@pytest.mark.parametrize(('inp_high, inp_low, level_low, level_high, mode, ref_scale, ref_zero_point'),
                         ((np.zeros((10, 10)), np.zeros((10, 10)), -128, 127, QuantizationMode.SYMMETRIC,
                           np.zeros((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (np.zeros((10, 10)), np.zeros((10, 10)), -128, 127, QuantizationMode.ASYMMETRIC,
                           np.zeros((10, 10)),
                           -128 * np.ones((10, 10), dtype=np.int32)),

                          (np.ones((10, 10)), np.zeros((10, 10)), 10, 999, QuantizationMode.SYMMETRIC,
                           0.00101112 * np.ones((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (np.ones((10, 10)), np.zeros((10, 10)), 10, 999, QuantizationMode.ASYMMETRIC,
                           0.001011122 * np.ones((10, 10)),
                           10 * np.ones((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), 0, 1000, QuantizationMode.SYMMETRIC,
                           0.011 * np.ones((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), 0, 1000, QuantizationMode.ASYMMETRIC,
                           0.011 * np.ones((10, 10)),
                           91 * np.ones((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), -10, -1, QuantizationMode.SYMMETRIC,
                           1.2222222 * np.ones((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), -10, -1, QuantizationMode.ASYMMETRIC,
                           1.22222222 * np.ones((10, 10)),
                           -9 * np.ones((10, 10), dtype=np.int32))
                          )
                         )
def test_calculate_scale_zero_point(inp_high, inp_low, level_low, level_high, mode, ref_scale, ref_zero_point):
    scale, zero_point = calculate_scale_zero_point(inp_low, inp_high, level_low, level_high, mode)
    assert np.allclose(ref_scale, scale)
    assert np.allclose(ref_zero_point, zero_point)


@pytest.mark.parametrize('num_bits, tensor_type, ref_levels', ((0, np.int8, (-1, -1)),
                                                               (2, np.int8, (-2, 1)),
                                                               (2, np.uint8, (0, 3)),
                                                               (8, np.int8, (-128, 127)),
                                                               (8, np.uint8, (0, 255)),
                                                               (10, np.int8, (-512, 511)),
                                                               (10, np.uint8, (0, 1023))))
def test_calculate_levels(num_bits, tensor_type, ref_levels):
    assert (ref_levels[0], ref_levels[1]) == get_level_low_level_high(tensor_type, num_bits)
