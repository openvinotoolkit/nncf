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
import numpy as np
from dataclasses import dataclass
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.experimental.onnx.quantization.quantizer_parameters import calculate_scale_zero_point
from nncf.experimental.onnx.quantization.quantizer_parameters import calculate_activation_quantizer_parameters
from nncf.experimental.onnx.quantization.quantizer_parameters import calculate_weight_quantizer_parameters
from nncf.experimental.onnx.quantization.quantizer_parameters import get_level_low_level_high
from nncf.experimental.onnx.quantization.quantizer_parameters import ONNXQuantizerLayerParameters
from nncf.experimental.onnx.statistics.statistics import ONNXMinMaxTensorStatistic


@dataclass
class CaseToTestScaleZP:
    max_val: np.ndarray
    min_val: np.ndarray
    level_low: int
    level_high: int
    mode: QuantizationMode
    ref_scale: np.ndarray
    ref_zero_point: np.ndarray


@pytest.mark.parametrize('case', (CaseToTestScaleZP(max_val=np.zeros((10, 10)),
                                                    min_val=np.zeros((10, 10)),
                                                    level_low=-128,
                                                    level_high=127,
                                                    mode=QuantizationMode.SYMMETRIC,
                                                    ref_scale=np.zeros((10, 10)),
                                                    ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=np.zeros((10, 10)),
                                                    min_val=np.zeros((10, 10)),
                                                    level_low=-128,
                                                    level_high=127,
                                                    mode=QuantizationMode.ASYMMETRIC,
                                                    ref_scale=np.zeros((10, 10)),
                                                    ref_zero_point=-128 * np.ones((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=np.ones((10, 10)),
                                                    min_val=np.zeros((10, 10)),
                                                    level_low=10,
                                                    level_high=999,
                                                    mode=QuantizationMode.SYMMETRIC,
                                                    ref_scale=0.00202224 * np.ones((10, 10)),
                                                    ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=np.ones((10, 10)),
                                                    min_val=np.zeros((10, 10)),
                                                    level_low=10,
                                                    level_high=999,
                                                    mode=QuantizationMode.ASYMMETRIC,
                                                    ref_scale=0.001011122 * np.ones((10, 10)),
                                                    ref_zero_point=10 * np.ones((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=10 * np.ones((10, 10)),
                                                    min_val=-np.ones((10, 10)),
                                                    level_low=0,
                                                    level_high=1000,
                                                    mode=QuantizationMode.SYMMETRIC,
                                                    ref_scale=0.02 * np.ones((10, 10)),
                                                    ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=10 * np.ones((10, 10)),
                                                    min_val=-np.ones((10, 10)),
                                                    level_low=0,
                                                    level_high=1000,
                                                    mode=QuantizationMode.ASYMMETRIC,
                                                    ref_scale=0.011 * np.ones((10, 10)),
                                                    ref_zero_point=91 * np.ones((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=10 * np.ones((10, 10)),
                                                    min_val=-np.ones((10, 10)),
                                                    level_low=-10,
                                                    level_high=-1,
                                                    mode=QuantizationMode.SYMMETRIC,
                                                    ref_scale=2.2222222 * np.ones((10, 10)),
                                                    ref_zero_point=np.zeros((10, 10), dtype=np.int32)),
                                  CaseToTestScaleZP(max_val=10 * np.ones((10, 10)),
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


@pytest.mark.parametrize('num_bits, tensor_type, ref_levels', ((0, np.int8, (-1, -1)),
                                                               (2, np.int8, (-2, 1)),
                                                               (2, np.uint8, (0, 3)),
                                                               (8, np.int8, (-128, 127)),
                                                               (8, np.uint8, (0, 255)),
                                                               (10, np.int8, (-512, 511)),
                                                               (10, np.uint8, (0, 1023))))
def test_calculate_levels(num_bits, tensor_type, ref_levels):
    assert (ref_levels[0], ref_levels[1]) == get_level_low_level_high(tensor_type, num_bits)


@dataclass
class CaseToTestActivationQParams:
    num_bits: int
    mode: QuantizationMode
    activations_signed: bool
    per_channel: bool
    axis: int
    ref_tensor_type: np.ndarray


CASES_FOR_TEST = (CaseToTestActivationQParams(num_bits=8,
                                              mode=QuantizationMode.SYMMETRIC,
                                              activations_signed=None,
                                              per_channel=False,
                                              axis=None,
                                              ref_tensor_type=np.int8),
                  CaseToTestActivationQParams(num_bits=8,
                                              mode=QuantizationMode.ASYMMETRIC,
                                              activations_signed=None,
                                              per_channel=False,
                                              axis=None,
                                              ref_tensor_type=np.int8),
                  CaseToTestActivationQParams(num_bits=8,
                                              mode=QuantizationMode.SYMMETRIC,
                                              activations_signed=None,
                                              per_channel=True,
                                              axis=1,
                                              ref_tensor_type=np.int8),
                  CaseToTestActivationQParams(num_bits=8,
                                              mode=QuantizationMode.SYMMETRIC,
                                              activations_signed=None,
                                              per_channel=True,
                                              axis=1,
                                              ref_tensor_type=np.int8),
                  )


@pytest.mark.parametrize('case_to_test', (CASES_FOR_TEST))
def test_calculate_activation_quantizer_parameters(case_to_test):
    statistics = ONNXMinMaxTensorStatistic(-1 * np.ones((3, 10, 10)), np.ones((3, 10, 10)))
    qconfig = QuantizerConfig(num_bits=case_to_test.num_bits,
                              mode=case_to_test.mode,
                              signedness_to_force=case_to_test.activations_signed,
                              per_channel=case_to_test.per_channel)
    ref_quantize_params = ONNXQuantizerLayerParameters(-1 * np.ones((3, 10, 10)), np.ones((3, 10, 10)),
                                                       mode=case_to_test.mode,
                                                       axis=case_to_test.axis,
                                                       tensor_type=case_to_test.ref_tensor_type)
    quantize_params = calculate_activation_quantizer_parameters(statistics, qconfig, case_to_test.axis)
    assert ref_quantize_params.mode == quantize_params.mode
    assert ref_quantize_params.axis == quantize_params.axis
    assert ref_quantize_params.tensor_type == quantize_params.tensor_type


@pytest.mark.parametrize('case_to_test', (CASES_FOR_TEST))
def test_calculate_weight_quantizer_parameters(case_to_test):
    qconfig = QuantizerConfig(num_bits=case_to_test.num_bits,
                              mode=case_to_test.mode,
                              signedness_to_force=case_to_test.activations_signed,
                              per_channel=case_to_test.per_channel)
    ref_quantize_params = ONNXQuantizerLayerParameters(-1 * np.ones((3, 10, 10)), np.ones((3, 10, 10)),
                                                       mode=case_to_test.mode,
                                                       axis=case_to_test.axis,
                                                       tensor_type=case_to_test.ref_tensor_type)
    quantize_params = calculate_weight_quantizer_parameters(np.ones((3, 10, 10)), qconfig, case_to_test.axis)
    assert ref_quantize_params.mode == quantize_params.mode
    assert ref_quantize_params.axis == quantize_params.axis
    assert ref_quantize_params.tensor_type == quantize_params.tensor_type
