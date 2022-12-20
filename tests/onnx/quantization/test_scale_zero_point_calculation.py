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

import pytest
from nncf.common.quantization.structs import QuantizationMode
from nncf.experimental.onnx.quantization.quantizer_parameters import calculate_scale_zero_point
import numpy as np
from dataclasses import dataclass


@dataclass
class CaseToTest:
    max_val: np.ndarray
    min_val: np.ndarray
    level_low: int
    level_high: int
    mode: QuantizationMode
    ref_scale: np.ndarray
    ref_zero_point: np.ndarray


@pytest.mark.parametrize('case', (CaseToTest(max_val=np.zeros((10, 10)),
                                             min_val=np.zeros((10, 10)),
                                             level_low=-128,
                                             level_high=127,
                                             mode=QuantizationMode.SYMMETRIC,
                                             ref_scale=np.zeros((10, 10)),
                                             ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=np.zeros((10, 10)),
                                             min_val=np.zeros((10, 10)),
                                             level_low=-128,
                                             level_high=127,
                                             mode=QuantizationMode.ASYMMETRIC,
                                             ref_scale=np.zeros((10, 10)),
                                             ref_zero_point=-128 * np.ones((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=np.ones((10, 10)),
                                             min_val=np.zeros((10, 10)),
                                             level_low=10,
                                             level_high=999,
                                             mode=QuantizationMode.SYMMETRIC,
                                             ref_scale=0.00202224 * np.ones((10, 10)),
                                             ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=np.ones((10, 10)),
                                             min_val=np.zeros((10, 10)),
                                             level_low=10,
                                             level_high=999,
                                             mode=QuantizationMode.ASYMMETRIC,
                                             ref_scale=0.001011122 * np.ones((10, 10)),
                                             ref_zero_point=10 * np.ones((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=10 * np.ones((10, 10)),
                                             min_val=-np.ones((10, 10)),
                                             level_low=0,
                                             level_high=1000,
                                             mode=QuantizationMode.SYMMETRIC,
                                             ref_scale=0.02 * np.ones((10, 10)),
                                             ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=10 * np.ones((10, 10)),
                                             min_val=-np.ones((10, 10)),
                                             level_low=0,
                                             level_high=1000,
                                             mode=QuantizationMode.ASYMMETRIC,
                                             ref_scale=0.011 * np.ones((10, 10)),
                                             ref_zero_point=91 * np.ones((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=10 * np.ones((10, 10)),
                                             min_val=-np.ones((10, 10)),
                                             level_low=-10,
                                             level_high=-1,
                                             mode=QuantizationMode.SYMMETRIC,
                                             ref_scale=2.2222222 * np.ones((10, 10)),
                                             ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTest(max_val=10 * np.ones((10, 10)),
                                             min_val=-np.ones((10, 10)),
                                             level_low=-10,
                                             level_high=-1,
                                             mode=QuantizationMode.ASYMMETRIC,
                                             ref_scale=1.22222222 * np.ones((10, 10)),
                                             ref_zero_point=-9 * np.ones((10, 10), dtype=np.int32))
                                  )
                         )
def test_calculate_scale_zero_point(case):
    assert np.allclose((case.ref_scale, case.ref_zero_point),
                       calculate_scale_zero_point(max_val=case.max_val, min_val=case.min_val,
                                                  level_low=case.level_low, level_high=case.level_high,
                                                  mode=case.mode))
