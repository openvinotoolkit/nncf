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

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pytest
import torch

from nncf.common.graph.transformations.commands import TargetType
from nncf.common.quantization.structs import QuantizationScheme as QuantizationMode
from nncf.common.quantization.structs import QuantizerGroup
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.experimental.quantization.quantizer import FXQuantizerConfig
from nncf.experimental.quantization.quantizer import IntDtype
from nncf.quantization.algorithms.min_max.torch_fx_backend import FXMinMaxAlgoBackend
from nncf.quantization.fake_quantize import calculate_quantizer_parameters
from nncf.tensor import Tensor

INPUT_SHAPE = (2, 3, 4, 5)


@dataclass
class CaseQuantParams:
    stat: MinMaxTensorStatistic
    per_channel: bool
    quant_group: QuantizerGroup
    ref_scale: np.ndarray
    narrow_range: bool


SYM_CASES = (
    CaseQuantParams(
        stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
        per_channel=False,
        narrow_range=False,
        quant_group=QuantizerGroup.ACTIVATIONS,
        ref_scale=0.00393701,
    ),
    CaseQuantParams(
        stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
        per_channel=False,
        narrow_range=True,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_scale=0.003937008,
    ),
    CaseQuantParams(
        stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, -0.4, -0.3])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
        per_channel=True,
        narrow_range=False,
        quant_group=QuantizerGroup.ACTIVATIONS,
        ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
    ),
    CaseQuantParams(
        stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, -0.4, -0.3])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
        per_channel=True,
        narrow_range=True,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_scale=torch.tensor([0.003937008, 0.0031496063, 0.0023622047]),
    ),
    CaseQuantParams(
        stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5])), Tensor(torch.tensor([0.5]))),
        per_channel=True,
        narrow_range=True,
        quant_group=QuantizerGroup.WEIGHTS,
        ref_scale=torch.tensor([0.003937008]),
    ),
)


@pytest.mark.parametrize("case_to_test", SYM_CASES)
@pytest.mark.parametrize("dtype", [IntDtype.UINT8, IntDtype.INT8])
def test_quantizer_params_sym(case_to_test: CaseQuantParams, dtype: Optional[IntDtype]):
    per_ch = case_to_test.per_channel
    narrow_range = case_to_test.narrow_range
    mode = QuantizationMode.SYMMETRIC
    signedness_to_force = None
    qconfig = FXQuantizerConfig(
        num_bits=8,
        mode=mode,
        per_channel=per_ch,
        narrow_range=narrow_range,
        signedness_to_force=signedness_to_force,
        dest_dtype=dtype,
    )

    quantizer = _get_quantizer(case_to_test, qconfig)
    assert quantizer.qscheme is torch.per_channel_symmetric if case_to_test.per_channel else torch.per_tensor_symmetric

    signed = signedness_to_force or dtype is IntDtype.INT8
    if signed:
        assert torch.allclose(quantizer.zero_point, torch.tensor(0, dtype=torch.int8))
    else:
        assert torch.allclose(quantizer.zero_point, torch.tensor(127 if narrow_range else 128, dtype=torch.uint8))

    scale = quantizer.scale.detach().numpy()
    assert np.allclose(scale, case_to_test.ref_scale)
    _check_q_min_q_max(quantizer, signed, narrow_range)


SYM_CASES_SIGNEDNESS_TO_FORSE = (
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(0.1)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00196078,
        ),
        False,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(0.1)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00393701,
        ),
        False,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00393701,
        ),
        True,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00393701,
        ),
        True,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([0.4, 0.3, 0.2])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.0019607844, 0.0015686274, 0.0011764707]),
        ),
        False,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([0.4, 0.3, 0.2])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
        ),
        False,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, 0.2, 0.1])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
        ),
        True,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, 0.2, 0.1])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
        ),
        True,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(0.1)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.001968504,
        ),
        False,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(0.1)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00393701,
        ),
        False,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00393701,
        ),
        True,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=0.00393701,
        ),
        True,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([0.4, 0.3, 0.2])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.0019685, 0.0015748, 0.0011811]),
        ),
        False,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([0.4, 0.3, 0.2])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
        ),
        False,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, 0.2, 0.1])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
        ),
        True,
        None,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, 0.2, 0.1])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701, 0.00314961, 0.0023622]),
        ),
        True,
        True,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5])), Tensor(torch.tensor([0.5]))),
            per_channel=True,
            narrow_range=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            ref_scale=torch.tensor([0.00393701]),
        ),
        True,
        True,
    ),
)


@pytest.mark.parametrize("case_to_test,ref_signed,signedness_to_force", SYM_CASES_SIGNEDNESS_TO_FORSE)
def test_quantizer_params_sym_nr(case_to_test: CaseQuantParams, ref_signed: bool, signedness_to_force: Optional[bool]):
    per_ch = case_to_test.per_channel
    narrow_range = case_to_test.narrow_range
    mode = QuantizationMode.SYMMETRIC
    qconfig = FXQuantizerConfig(
        num_bits=8,
        mode=mode,
        per_channel=per_ch,
        narrow_range=narrow_range,
        signedness_to_force=signedness_to_force,
        dest_dtype=None,
    )

    quantizer = _get_quantizer(case_to_test, qconfig)
    assert quantizer.qscheme is torch.per_channel_symmetric if case_to_test.per_channel else torch.per_tensor_symmetric

    signed = signedness_to_force or ref_signed

    assert torch.allclose(quantizer.zero_point, torch.tensor(0, dtype=torch.int8 if signed else torch.uint8))

    scale = quantizer.scale.detach().numpy()
    assert np.allclose(scale, case_to_test.ref_scale)

    _check_q_min_q_max(quantizer, signed, narrow_range)


ASYM_CASES = (
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            quant_group=QuantizerGroup.WEIGHTS,
            narrow_range=True,
            ref_scale=0.00395251,
        ),
        0.0,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor(-0.5)), Tensor(torch.tensor(0.5))),
            per_channel=False,
            quant_group=QuantizerGroup.ACTIVATIONS,
            narrow_range=False,
            ref_scale=0.00393701,
        ),
        0.0,
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, -0.2, 0.1])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            quant_group=QuantizerGroup.ACTIVATIONS,
            narrow_range=False,
            ref_scale=torch.tensor([0.00393701, 0.00235294, 0.00117647]),
        ),
        [0, -43, -128],
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5, -0.2, 0.1])), Tensor(torch.tensor([0.5, 0.4, 0.3]))),
            per_channel=True,
            quant_group=QuantizerGroup.WEIGHTS,
            narrow_range=True,
            ref_scale=torch.tensor([0.00395251, 0.0023622, 0.0011811]),
        ),
        [0, -42, -127],
    ),
    (
        CaseQuantParams(
            stat=MinMaxTensorStatistic(Tensor(torch.tensor([-0.5])), Tensor(torch.tensor([0.5]))),
            per_channel=True,
            quant_group=QuantizerGroup.WEIGHTS,
            narrow_range=True,
            ref_scale=torch.tensor([0.00395251]),
        ),
        [0],
    ),
)


@pytest.mark.parametrize("case_to_test,ref_zp", ASYM_CASES)
@pytest.mark.parametrize("dtype", [IntDtype.UINT8, IntDtype.INT8])
def test_quantizer_params_asym(case_to_test: CaseQuantParams, ref_zp: Union[int, list[int]], dtype: Optional[IntDtype]):
    per_ch = case_to_test.per_channel
    narrow_range = case_to_test.narrow_range
    mode = QuantizationMode.ASYMMETRIC
    qconfig = FXQuantizerConfig(
        num_bits=8,
        mode=mode,
        per_channel=per_ch,
        narrow_range=narrow_range,
        signedness_to_force=None,
        dest_dtype=dtype,
    )

    quantizer = _get_quantizer(case_to_test, qconfig)
    assert quantizer.qscheme is torch.per_channel_affine if case_to_test.per_channel else torch.per_tensor_affine

    signed = dtype is IntDtype.INT8
    ref_zp = torch.tensor(ref_zp)
    if not signed:
        ref_zp += 127 if narrow_range else 128
    assert torch.allclose(quantizer.zero_point, ref_zp.to(dtype=torch.int8 if signed else torch.uint8))

    scale = quantizer.scale.detach().numpy()
    assert np.allclose(scale, case_to_test.ref_scale)

    _check_q_min_q_max(quantizer, signed, narrow_range)


def _get_quantizer(case_to_test: CaseQuantParams, qconfig: FXQuantizerConfig):
    fq_params = calculate_quantizer_parameters(case_to_test.stat, qconfig, case_to_test.quant_group, half_range=False)

    ch_axis = 1 if case_to_test.per_channel and case_to_test.quant_group == QuantizerGroup.WEIGHTS else 0
    target_type = (
        TargetType.OPERATION_WITH_WEIGHTS
        if case_to_test.quant_group == QuantizerGroup.WEIGHTS
        else TargetType.PRE_LAYER_OPERATION
    )
    quantizer = FXMinMaxAlgoBackend._create_quantizer(qconfig, ch_axis, fq_params, target_type)

    assert quantizer.ch_axis == ch_axis

    return quantizer


def _check_q_min_q_max(quantizer, signed, narrow_range):
    if signed:
        ref_quant_min = -127 if narrow_range else -128
        ref_quant_max = 127
    else:
        ref_quant_min = 0
        ref_quant_max = 254 if narrow_range else 255

    assert quantizer.quant_min == ref_quant_min
    assert quantizer.quant_max == ref_quant_max
