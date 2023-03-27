
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
from dataclasses import dataclass
import numpy as np

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizationMode
from nncf.common.quantization.structs import QuantizerGroup
from nncf.quantization.algorithms.min_max.torch_backend import PTMinMaxAlgoBackend
from nncf.quantization.fake_quantize import FakeQuantizeParameters

# pylint: disable=protected-access

INPUT_SHAPE = (2, 3, 4, 5)


@dataclass
class CaseSymParams:
    fq_params: FakeQuantizeParameters
    per_channel: bool
    quant_group: QuantizerGroup
    ref_scale: np.ndarray


SYM_CASES = (CaseSymParams(fq_params=FakeQuantizeParameters(
                                     np.array(-0.49920455, dtype=np.float32),
                                     np.array(0.49530452, dtype=np.float32),
                                     np.array(-0.49920455, dtype=np.float32),
                                     np.array(0.49530452, dtype=np.float32),
                                     256),
                           per_channel=False,
                           quant_group=QuantizerGroup.ACTIVATIONS,
                           ref_scale=0.49530452),
             CaseSymParams(fq_params=FakeQuantizeParameters(
                                     np.array(-0.49530452, dtype=np.float32),
                                     np.array(0.49530452, dtype=np.float32),
                                     np.array(-0.49530452, dtype=np.float32),
                                     np.array(0.49530452, dtype=np.float32),
                                     255),
                           per_channel=False,
                           quant_group=QuantizerGroup.WEIGHTS,
                           ref_scale=0.49530452),
             CaseSymParams(fq_params=FakeQuantizeParameters(
                                      np.array([-0.4835594, -0.49530452, -0.49221927], dtype=np.float32).reshape(1, 3, 1, 1),
                                      np.array([0.4797816, 0.49920455, 0.48837382], dtype=np.float32).reshape(1, 3, 1, 1),
                                      np.array([-0.4835594, -0.49530452, -0.49221927], dtype=np.float32).reshape(1, 3, 1, 1),
                                      np.array([0.4797816, 0.49920455, 0.48837382], dtype=np.float32).reshape(1, 3, 1, 1),
                                      256),
                            per_channel=True,
                            quant_group=QuantizerGroup.ACTIVATIONS,
                            ref_scale=np.array([0.4797816, 0.49920455, 0.48837382]).reshape(1, 3, 1, 1)),
             CaseSymParams(fq_params=FakeQuantizeParameters(
                                     np.array([-0.48837382, -0.49530452], dtype=np.float32).reshape(2, 1, 1, 1),
                                     np.array([0.48837382, 0.49530452], dtype=np.float32).reshape(2, 1, 1, 1),
                                     np.array([-0.48837382, -0.49530452], dtype=np.float32).reshape(2, 1, 1, 1),
                                     np.array([0.48837382, 0.49530452], dtype=np.float32).reshape(2, 1, 1, 1),
                                     255),
                           per_channel=True,
                           quant_group=QuantizerGroup.WEIGHTS,
                           ref_scale=np.array([0.48837382, 0.49530452]).reshape(2, 1, 1, 1)),
)


@pytest.mark.parametrize('case_to_test', SYM_CASES)
def test_quantizer_params_sym(case_to_test):
    per_ch = case_to_test.per_channel
    fq_params = case_to_test.fq_params
    quant_group = case_to_test.quant_group
    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.SYMMETRIC, per_channel=per_ch)

    if not per_ch:
        scale_shape = [1]
    else:
        scale_shape = [1] * len(INPUT_SHAPE)
        axis = 0 if quant_group == QuantizerGroup.WEIGHTS else 1
        scale_shape[axis] = INPUT_SHAPE[axis]

    target_type = TargetType.OPERATION_WITH_WEIGHTS if quant_group == QuantizerGroup.WEIGHTS \
         else TargetType.PRE_LAYER_OPERATION
    quantizer = PTMinMaxAlgoBackend._create_quantizer(qconfig, scale_shape, fq_params, target_type)

    assert quantizer.levels == fq_params.levels
    scale = quantizer.scale.detach().numpy()
    ref_scale = case_to_test.ref_scale
    assert np.allclose(scale, ref_scale)


@dataclass
class CaseAsymParams:
    fq_params: FakeQuantizeParameters
    per_channel: bool
    quant_group: QuantizerGroup
    ref_inp_low: np.ndarray
    ref_inp_range: np.ndarray


ASYM_CASES = (CaseAsymParams(fq_params=FakeQuantizeParameters(
                                       np.array(-0.49530452, dtype=np.float32),
                                       np.array(0.49143496, dtype=np.float32),
                                       np.array(-0.49530452, dtype=np.float32),
                                       np.array(0.49143496, dtype=np.float32),
                                       256),
                             per_channel=False,
                             quant_group=QuantizerGroup.WEIGHTS,
                             ref_inp_low=-0.49530452,
                             ref_inp_range=0.98673948),
              CaseAsymParams(fq_params=FakeQuantizeParameters(
                                       np.array(-0.49530452, dtype=np.float32),
                                       np.array(0.49143496, dtype=np.float32),
                                       np.array(-0.49530452, dtype=np.float32),
                                       np.array(0.49143496, dtype=np.float32),
                                       256),
                             per_channel=False,
                             quant_group=QuantizerGroup.ACTIVATIONS,
                             ref_inp_low=-0.49530452,
                             ref_inp_range=0.98673948),
              CaseAsymParams(fq_params=FakeQuantizeParameters(
                                       np.array([-0.48051512, -0.49776307, -0.44099426], dtype=np.float32).reshape(1, 3, 1, 1),
                                       np.array([0.4767611, 0.47861832, 0.48837382], dtype=np.float32).reshape(1, 3, 1, 1),
                                       np.array([-0.48051512, -0.49776307, -0.44099426], dtype=np.float32).reshape(1, 3, 1, 1),
                                       np.array([0.4767611, 0.47861832, 0.48837382], dtype=np.float32).reshape(1, 3, 1, 1),
                                       256),
                             per_channel=True,
                             quant_group=QuantizerGroup.ACTIVATIONS,
                             ref_inp_low=np.array([-0.48051512, -0.49776307, -0.44099426]).reshape(1, 3, 1, 1),
                             ref_inp_range=np.array([0.9572762, 0.9763814, 0.9293681]).reshape(1, 3, 1, 1)),
              CaseAsymParams(fq_params=FakeQuantizeParameters(
                                       np.array([-0.4845584, -0.49583155], dtype=np.float32).reshape(2, 1, 1, 1),
                                       np.array([0.48837382, 0.4767611], dtype=np.float32).reshape(2, 1, 1, 1),
                                       np.array([-0.4845584, -0.49583155], dtype=np.float32).reshape(2, 1, 1, 1),
                                       np.array([0.48837382, 0.4767611], dtype=np.float32).reshape(2, 1, 1, 1),
                                       256),
                             per_channel=True,
                             quant_group=QuantizerGroup.WEIGHTS,
                             ref_inp_low=np.array([-0.4845584, -0.49583155]).reshape(2, 1, 1, 1),
                             ref_inp_range=np.array([0.97293222, 0.97259265]).reshape(2, 1, 1, 1)),
)


@pytest.mark.parametrize('case_to_test', ASYM_CASES)
def test_quantizer_params_asym(case_to_test):
    per_ch = case_to_test.per_channel
    fq_params = case_to_test.fq_params
    quant_group = case_to_test.quant_group
    qconfig = QuantizerConfig(num_bits=8, mode=QuantizationMode.ASYMMETRIC, per_channel=per_ch)

    if not per_ch:
        scale_shape = [1]
    else:
        scale_shape = [1] * len(INPUT_SHAPE)
        axis = 0 if quant_group == QuantizerGroup.WEIGHTS else 1
        scale_shape[axis] = INPUT_SHAPE[axis]

    target_type = TargetType.OPERATION_WITH_WEIGHTS if quant_group == QuantizerGroup.WEIGHTS \
         else TargetType.PRE_LAYER_OPERATION
    quantizer = PTMinMaxAlgoBackend._create_quantizer(qconfig, scale_shape, fq_params, target_type)
    assert quantizer.levels == fq_params.levels
    assert np.allclose(quantizer.input_low.detach().numpy(), case_to_test.ref_inp_low)
    assert np.allclose(quantizer.input_range.detach().numpy(), case_to_test.ref_inp_range)
