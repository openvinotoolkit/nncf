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
from nncf.common.quantization.structs import QuantizerConfig
from nncf.quantization.algorithms.min_max.quantizer_parameters import QuantizerLayerParameters
from nncf.onnx.quantization.quantizer_parameters import calculate_scale_zero_point
from nncf.onnx.quantization.quantizer_parameters import get_level_low_level_high


@pytest.mark.parametrize(('max_val, min_val, level_low, level_high, mode, ref_scale, ref_zero_point'),
                         ((np.zeros((10, 10)), np.zeros((10, 10)), -128, 127, QuantizationMode.SYMMETRIC,
                           np.zeros((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (np.zeros((10, 10)), np.zeros((10, 10)), -128, 127, QuantizationMode.ASYMMETRIC,
                           np.zeros((10, 10)),
                           -128 * np.ones((10, 10), dtype=np.int32)),

                          (np.ones((10, 10)), np.zeros((10, 10)), 10, 999, QuantizationMode.SYMMETRIC,
                           0.00202224 * np.ones((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (np.ones((10, 10)), np.zeros((10, 10)), 10, 999, QuantizationMode.ASYMMETRIC,
                           0.001011122 * np.ones((10, 10)),
                           10 * np.ones((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), 0, 1000, QuantizationMode.SYMMETRIC,
                           0.02 * np.ones((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), 0, 1000, QuantizationMode.ASYMMETRIC,
                           0.011 * np.ones((10, 10)),
                           91 * np.ones((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), -10, -1, QuantizationMode.SYMMETRIC,
                           2.2222222 * np.ones((10, 10)),
                           np.zeros((10, 10), dtype=np.int32)),

                          (10 * np.ones((10, 10)), -np.ones((10, 10)), -10, -1, QuantizationMode.ASYMMETRIC,
                           1.22222222 * np.ones((10, 10)),
                           -9 * np.ones((10, 10), dtype=np.int32))
                          )
                         )
@pytest.mark.parametrize('tensor_type', [np.int8, np.uint8])
def test_calculate_scale_zero_point(max_val, min_val, level_low, level_high, mode, ref_scale, ref_zero_point,
                                    tensor_type):
    ref_scale_ = ref_scale.copy()
    if tensor_type == np.uint8 and mode == QuantizationMode.SYMMETRIC:
        ref_scale_ /= 2

    num_bits = np.ceil(np.log2(np.abs(level_high - level_low)))
    params = QuantizerLayerParameters(min_val, max_val, min_val, max_val, 2 ** num_bits)
    qconfig = QuantizerConfig(num_bits=num_bits, mode=mode, signedness_to_force=level_low < 0, per_channel=None)
    assert np.allclose((ref_scale_, ref_zero_point), calculate_scale_zero_point(params, qconfig))


@pytest.mark.parametrize('num_bits, tensor_type, ref_levels', ((0, np.int8, (-1, -1)),
                                                               (2, np.int8, (-2, 1)),
                                                               (2, np.uint8, (0, 3)),
                                                               (8, np.int8, (-128, 127)),
                                                               (8, np.uint8, (0, 255)),
                                                               (10, np.int8, (-512, 511)),
                                                               (10, np.uint8, (0, 1023))))
def test_calculate_levels(num_bits, tensor_type, ref_levels):
    assert (ref_levels[0], ref_levels[1]) == get_level_low_level_high(tensor_type, num_bits)
