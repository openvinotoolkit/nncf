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

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import nncf
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.quantization.fake_quantize import FakeQuantizeParameters
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.tensor import functions as fns
from tests.cross_fw.shared.json import dump_to_json
from tests.cross_fw.shared.json import load_json

FQ_CALCULATED_PARAMETERS_PATH = Path(__file__).parent / "fq_params" / "fq_params.json"


def compare_fq_parameters(ref_params, params):
    assert ref_params.levels == params.levels
    assert ref_params.input_low.shape == params.input_low.shape
    assert ref_params.input_high.shape == params.input_high.shape
    assert ref_params.output_low.shape == params.output_low.shape
    assert ref_params.output_high.shape == params.output_high.shape
    assert fns.allclose(ref_params.input_low, params.input_low)
    assert fns.allclose(ref_params.input_high, params.input_high)
    assert fns.allclose(ref_params.output_low, params.output_low)
    assert fns.allclose(ref_params.output_high, params.output_high)


def get_test_reference_key(q_group, q_config, narrow_range, hf_range):
    mode = q_config.mode
    sign = q_config.signedness_to_force
    per_ch = q_config.per_channel
    return f"{q_group.value}_{mode}_sign_{sign}_per_ch_{per_ch}_narrow_range_{narrow_range}_hf_range_{hf_range}"


def read_ref_fq_params(q_group, q_config, narrow_range, hf_range):
    fq_params = load_json(FQ_CALCULATED_PARAMETERS_PATH)
    key = get_test_reference_key(q_group, q_config, narrow_range, hf_range)
    inp_l = np.array(fq_params[key]["input_low"]).astype(np.float32)
    inp_h = np.array(fq_params[key]["input_high"]).astype(np.float32)
    out_l = np.array(fq_params[key]["output_low"]).astype(np.float32)
    out_h = np.array(fq_params[key]["output_high"]).astype(np.float32)
    levels = fq_params[key]["levels"]
    ref_quantize_params = FakeQuantizeParameters(inp_l, inp_h, out_l, out_h, levels)
    return ref_quantize_params


def dump_fq_params(fq_params, q_group, q_config, narrow_range, hf_range):
    key = get_test_reference_key(q_group, q_config, narrow_range, hf_range)
    all_fq_params = load_json(FQ_CALCULATED_PARAMETERS_PATH)
    fq_params_dict = parse_fq_params_to_dict(fq_params)
    all_fq_params[key] = fq_params_dict
    dump_to_json(FQ_CALCULATED_PARAMETERS_PATH, all_fq_params)


def parse_fq_params_to_dict(fq_params):
    return {
        "levels": fq_params.levels,
        "input_low": fq_params.input_low,
        "input_high": fq_params.input_high,
        "output_low": fq_params.output_low,
        "output_high": fq_params.output_high,
    }


@dataclass
class CaseFQParams:
    q_config: QuantizerConfig
    q_group: QuantizerGroup
    narrow_range: bool
    half_range: bool
    should_fail: bool


TO_TEST = [
    # WEIGHT QUANTIZER CONFIGURATIONS
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
        q_group=QuantizerGroup.WEIGHTS,
        narrow_range=True,
        half_range=True,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
        q_group=QuantizerGroup.WEIGHTS,
        narrow_range=True,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
        q_group=QuantizerGroup.WEIGHTS,
        narrow_range=False,
        half_range=True,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC),
        q_group=QuantizerGroup.WEIGHTS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC),
        q_group=QuantizerGroup.WEIGHTS,
        narrow_range=False,
        half_range=True,
        should_fail=True,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC),
        q_group=QuantizerGroup.WEIGHTS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    # ACTIVATION QUANTIZER CONFIGURATIONS
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=False),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=True),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=False),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=True),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(
            num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=False
        ),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(
            num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=True
        ),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=False,
        should_fail=False,
    ),
    CaseFQParams(
        q_config=QuantizerConfig(
            num_bits=8, mode=QuantizationMode.ASYMMETRIC, signedness_to_force=True, per_channel=True
        ),
        q_group=QuantizerGroup.ACTIVATIONS,
        narrow_range=False,
        half_range=True,
        should_fail=True,
    ),
]


class TemplateTestFQParams(ABC):
    @abstractmethod
    def to_nncf_tensor(self, t: np.array):
        raise NotImplementedError

    @pytest.mark.parametrize("case_to_test", TO_TEST)
    def test_calculate_quantizer_parameters(self, case_to_test):
        q_config = case_to_test.q_config
        quant_group = case_to_test.q_group
        narrow_range = case_to_test.narrow_range
        half_range = case_to_test.half_range

        rng = np.random.default_rng(0)
        data = rng.uniform(0, 1, (2, 3, 4, 5))

        if q_config.per_channel:
            axes = tuple(range(1, len(data.shape)))  # channel_axis = 0
        else:
            axes = None
        min_values = np.amin(data, axis=axes, keepdims=q_config.per_channel)
        if q_config.mode == QuantizationMode.SYMMETRIC:
            max_values = np.amax(np.abs(data), axis=axes, keepdims=q_config.per_channel)
        else:
            max_values = np.amax(data, axis=axes, keepdims=q_config.per_channel)

        statistics = MinMaxTensorStatistic(
            min_values=self.to_nncf_tensor(min_values),
            max_values=self.to_nncf_tensor(max_values),
        )
        if not case_to_test.should_fail:
            fq_params = calculate_quantizer_parameters(statistics, q_config, quant_group, narrow_range, half_range)
            # Uncomment lines below to generate reference for new models.
            # dump_fq_params(fq_params, quant_group, q_config, narrow_range, half_range)
            ref_fq_params = read_ref_fq_params(quant_group, q_config, narrow_range, half_range)
            compare_fq_parameters(fq_params, ref_fq_params)
        else:
            with pytest.raises(nncf.ValidationError):
                calculate_quantizer_parameters(statistics, q_config, quant_group, narrow_range, half_range)
