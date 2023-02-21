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
from dataclasses import dataclass

from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.experimental.openvino_native.quantization.quantizer_parameters import calculate_quantizer_parameters
from nncf.experimental.openvino_native.quantization.quantizer_parameters import OVQuantizerLayerParameters
from nncf.experimental.openvino_native.quantization.quantizer_parameters import get_weight_stats_shape
from nncf.experimental.openvino_native.statistics.statistics import OVMinMaxTensorStatistic
from tests.openvino.conftest import OPENVINO_NATIVE_TEST_ROOT
from tests.openvino.native.common import load_json

FQ_CALCULATED_PARAMETERS_PATH = OPENVINO_NATIVE_TEST_ROOT / 'data' / 'reference_scales' / 'fq_params_synthetic.json'


@dataclass
class CaseFQParams:
    mode: QuantizationMode
    signedness_to_force: bool
    per_channel: bool


CASES_FOR_TEST = (CaseFQParams(mode=QuantizationMode.SYMMETRIC,
                               signedness_to_force=False,
                               per_channel=False),
                  CaseFQParams(mode=QuantizationMode.SYMMETRIC,
                               signedness_to_force=False,
                               per_channel=True),
                  CaseFQParams(mode=QuantizationMode.ASYMMETRIC,
                               signedness_to_force=False,
                               per_channel=False),
                  CaseFQParams(mode=QuantizationMode.ASYMMETRIC,
                               signedness_to_force=False,
                               per_channel=True),
                  CaseFQParams(mode=QuantizationMode.SYMMETRIC,
                               signedness_to_force=True,
                               per_channel=False),
                  CaseFQParams(mode=QuantizationMode.SYMMETRIC,
                               signedness_to_force=True,
                               per_channel=True),
                  )


def compare_fq_parameters(ref_params, params):
    assert ref_params.levels == params.levels
    assert ref_params.input_low.shape == params.input_low.shape
    assert ref_params.input_high.shape == params.input_high.shape
    assert ref_params.output_low.shape == params.output_low.shape
    assert ref_params.output_high.shape == params.output_high.shape
    assert np.allclose(ref_params.input_low, params.input_low)
    assert np.allclose(ref_params.input_high, params.input_high)
    assert np.allclose(ref_params.output_low, params.output_low)
    assert np.allclose(ref_params.output_high, params.output_high)


def parse_test_data(stat_type, mode, sign, per_ch):
    fq_params = load_json(FQ_CALCULATED_PARAMETERS_PATH)
    input_data = np.array(fq_params['data']).astype(np.float32)
    key = f'{stat_type}_{mode}_sign_{sign}_per_ch_{per_ch}'
    inp_l = np.array(fq_params[key]['input_low']).astype(np.float32)
    inp_h = np.array(fq_params[key]['input_high']).astype(np.float32)
    out_l = np.array(fq_params[key]['output_low']).astype(np.float32)
    out_h = np.array(fq_params[key]['output_high']).astype(np.float32)
    levels = fq_params[key]['levels']
    ref_quantize_params = OVQuantizerLayerParameters(inp_l, inp_h, out_l, out_h, levels)
    return input_data, ref_quantize_params


@pytest.mark.parametrize('case_to_test', CASES_FOR_TEST)
def test_calculate_activation_quantizer_parameters(case_to_test):
    stat_type = 'activation'
    mode = case_to_test.mode
    sign = case_to_test.signedness_to_force
    per_ch = case_to_test.per_channel
    data, ref_quantize_params = parse_test_data(stat_type, mode, sign, per_ch)

    axes = (0, 2, 3) if case_to_test.per_channel else None
    min_values = np.amin(data, axes, keepdims=True)
    max_values = np.amax(np.abs(data), axes, keepdims=True)

    statistics = OVMinMaxTensorStatistic(min_values, max_values)
    qconfig = QuantizerConfig(num_bits=8, mode=mode, signedness_to_force=sign, per_channel=per_ch)
    quantize_params = calculate_quantizer_parameters(statistics, qconfig, QuantizerGroup.ACTIVATIONS)

    compare_fq_parameters(ref_quantize_params, quantize_params)


@pytest.mark.parametrize('case_to_test', CASES_FOR_TEST[:4])
def test_calculate_weight_quantizer_parameters(case_to_test):
    stat_type = 'weights'
    mode = case_to_test.mode
    sign = case_to_test.signedness_to_force
    per_ch = case_to_test.per_channel
    data, ref_quantize_params = parse_test_data(stat_type, mode, sign, per_ch)

    qconfig = QuantizerConfig(num_bits=8, mode=mode, signedness_to_force=sign, per_channel=per_ch)
    axes = None
    if qconfig.per_channel:
        bounds_shape = get_weight_stats_shape(data.shape, None)
        axes = tuple(i for i, dim in enumerate(bounds_shape) if dim == 1)

    min_values = np.amin(data, axis=axes, keepdims=qconfig.per_channel)
    max_values = np.amax(np.abs(data), axis=axes, keepdims=qconfig.per_channel)
    statistics = OVMinMaxTensorStatistic(min_values, max_values)
    quantize_params = calculate_quantizer_parameters(statistics, qconfig, QuantizerGroup.WEIGHTS)

    compare_fq_parameters(ref_quantize_params, quantize_params)
