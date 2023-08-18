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

from nncf.common.quantization.structs import QuantizerConfig
from nncf.common.quantization.structs import QuantizerGroup
from nncf.experimental.tensor import Tensor
from nncf.experimental.tensor import functions
from nncf.quantization.fake_quantize import asymmetric_range
from nncf.quantization.fake_quantize import fix_zero_filters_asymmetric
from nncf.quantization.fake_quantize import fix_zero_filters_symmetric
from nncf.quantization.fake_quantize import symmetric_range
from nncf.quantization.fake_quantize import tune_range


@pytest.mark.parametrize(
    "min_values, max_values, ref_low, ref_high",
    (
        (-1.1, 1.0, -1.1, 1.0),
        (0.1, 0.1000001, 0.0992, 0.1008001),
        ([0.1, -0.1], [0.1000001, 2.0], [0.0992, -0.1], [0.1008001, 2.0]),
    ),
)
def test_fix_zero_filters_asymmetric(min_values, max_values, ref_low, ref_high):
    level_low, level_high = fix_zero_filters_asymmetric(Tensor(np.array(min_values)), Tensor(np.array(max_values)))

    for val, ref in zip([level_low, level_high], [ref_low, ref_high]):
        if isinstance(ref, list):
            assert functions.all(functions.isclose(val, ref)), f"{val=}"
        else:
            assert functions.isclose(val, ref), f"{val=}"


@pytest.mark.parametrize(
    "max_values, ref",
    (
        (1.0, 1.0),
        (8e-7, 8e-05),
        ([1.0, 0.0], [1.0, 0.01]),
        ([[1.0, 0.0], [1.0, 2.0]], [[1.0, 0.02], [1.0, 2.0]]),
    ),
)
def test_fix_zero_filters_symmetric(max_values, ref):
    res = fix_zero_filters_symmetric(Tensor(np.array(max_values)))

    if isinstance(ref, list):
        assert functions.all(functions.isclose(res, ref))
    else:
        assert functions.isclose(res, ref)


@pytest.mark.parametrize(
    "left_border, right_border, unify_zp, ref_ra, ref_rb",
    (
        (-1.0, 1.0, True, -1.0078740157480315, 1.0),
        (-1.0, 1.0, False, -1.0078740157480315, 1.0),
        (-1.0, 1.1, True, -1.0, 1.1074380165289257),
        (-1.0, 1.1, False, -1.0, 1.1074380165289257),
        ([-1.0, 1.2], [1.0, 2.0], True, [-1.0, 0.66492147], [1.0, 2.0]),
        ([-1.0, 1.2], [1.0, 2.0], False, [-1.00787402, 1.19937206], [1.0, 2.0]),
    ),
)
def test_tune_range(left_border, right_border, unify_zp, ref_ra, ref_rb):
    ra, rb = tune_range(
        Tensor(np.array(left_border)),
        Tensor(np.array(right_border)),
        8,
        unify_zp,
    )

    for val, ref in zip([ra, rb], [ref_ra, ref_rb]):
        if isinstance(ref, list):
            assert functions.all(functions.isclose(val, ref))
        else:
            assert functions.isclose(val, ref)


@pytest.mark.parametrize(
    "min_values, max_values, levels, quantizer_config, q_group, ref_low, ref_high",
    (
        (-1.0, 1.0, 255, QuantizerConfig(8), QuantizerGroup.ACTIVATIONS, -1.0079051, 1.0),
        (
            [-1.0, 0.1],
            [1.0, 2.0],
            255,
            QuantizerConfig(8),
            QuantizerGroup.ACTIVATIONS,
            [-1.0079051, -2.0158103],
            [1.0, 2.0],
        ),
        (
            [-1.0, 0.1],
            [1.0, 2.0],
            255,
            QuantizerConfig(8),
            QuantizerGroup.WEIGHTS,
            [-1.0, -2.0],
            [1.0, 2.0],
        ),
        (
            [-1.0, 0.1],
            [1.0, 2.0],
            256,
            QuantizerConfig(8, signedness_to_force=True),
            QuantizerGroup.ACTIVATIONS,
            [-1.007874, -2.015748],
            [1.0, 2.0],
        ),
        (
            [-1.0, 0.1],
            [1.0, 2.0],
            256,
            QuantizerConfig(8, signedness_to_force=True),
            QuantizerGroup.WEIGHTS,
            [-1.0, -2.0],
            [1.0, 2.0],
        ),
    ),
)
def test_symmetric_range(min_values, max_values, levels, quantizer_config, q_group, ref_low, ref_high):
    level_low, level_high = symmetric_range(
        Tensor(np.array(min_values)),
        Tensor(np.array(max_values)),
        levels,
        quantizer_config,
        q_group,
    )
    for val, ref in zip([level_low, level_high], [ref_low, ref_high]):
        if isinstance(ref, list):
            assert functions.all(functions.isclose(val, ref)), f"{val=}"
        else:
            assert functions.isclose(val, ref), f"{val=}"


@pytest.mark.parametrize(
    "min_values, max_values, quantizer_config, q_group, unify_zp, ref_low, ref_high",
    (
        (-1.0, 1.0, QuantizerConfig(8), QuantizerGroup.ACTIVATIONS, False, -1.007874, 1.0),
        (-1.0, 1.0, QuantizerConfig(8), QuantizerGroup.WEIGHTS, False, -1.007874, 1.0),
        (0.1, 1.0, QuantizerConfig(8), QuantizerGroup.WEIGHTS, True, 0.0, 1.0),
        ([-1.0, 0.1], [1.0, 2.0], QuantizerConfig(8), QuantizerGroup.ACTIVATIONS, False, [-1.007874, 0.0], [1.0, 2.0]),
        ([-1.0, 0.1], [1.0, 2.0], QuantizerConfig(8), QuantizerGroup.WEIGHTS, False, [-1.007874, 0.0], [1.0, 2.0]),
        (
            [-1.0, 0.1],
            [1.0, 2.0],
            QuantizerConfig(8),
            QuantizerGroup.WEIGHTS,
            True,
            [-1.0, -0.6701571],
            [2.984375, 2.0],
        ),
        (
            [[-1.0], [0.1]],
            [[1.0], [2.0]],
            QuantizerConfig(8),
            QuantizerGroup.ACTIVATIONS,
            False,
            [[-1.007874], [0.0]],
            [[1.0], [2.0]],
        ),
    ),
)
def test_asymmetric_range(min_values, max_values, quantizer_config, q_group, unify_zp, ref_low, ref_high):
    level_low, level_high = asymmetric_range(
        Tensor(np.array(min_values)),
        Tensor(np.array(max_values)),
        quantizer_config,
        q_group,
        unify_zp,
    )
    for val, ref in zip([level_low, level_high], [ref_low, ref_high]):
        if isinstance(ref, list):
            assert functions.all(functions.isclose(val, ref)), f"{val=}"
        else:
            assert functions.isclose(val, ref), f"{val=}"
