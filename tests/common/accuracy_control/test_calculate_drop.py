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

import pytest

from nncf.parameters import DropType
from nncf.quantization.algorithms.accuracy_control.algorithm import calculate_accuracy_drop


@dataclass
class AccuracyDropTestCase:
    initial_metric: float
    quantized_metric: float
    drop_type: DropType
    expected_should_terminate: bool
    expected_accuracy_drop: float
    max_drop: float = 0.01


@pytest.mark.parametrize(
    "ts",
    [
        # ABSOLUTE
        AccuracyDropTestCase(
            initial_metric=0.2923,
            quantized_metric=0.3185,
            drop_type=DropType.ABSOLUTE,
            expected_should_terminate=True,
            expected_accuracy_drop=-0.0262,
        ),
        AccuracyDropTestCase(
            initial_metric=0.3185,
            quantized_metric=0.2923,
            drop_type=DropType.ABSOLUTE,
            expected_should_terminate=False,
            expected_accuracy_drop=0.0262,
        ),
        AccuracyDropTestCase(
            initial_metric=-0.2923,
            quantized_metric=-0.3185,
            drop_type=DropType.ABSOLUTE,
            expected_should_terminate=False,
            expected_accuracy_drop=0.0262,
        ),
        AccuracyDropTestCase(
            initial_metric=-0.3185,
            quantized_metric=-0.2923,
            drop_type=DropType.ABSOLUTE,
            expected_should_terminate=True,
            expected_accuracy_drop=-0.0262,
        ),
        # RELATIVE
        AccuracyDropTestCase(
            initial_metric=0.2923,
            quantized_metric=0.3185,
            drop_type=DropType.RELATIVE,
            expected_should_terminate=True,
            expected_accuracy_drop=None,
        ),
        AccuracyDropTestCase(
            initial_metric=0.3185,
            quantized_metric=0.2923,
            drop_type=DropType.RELATIVE,
            expected_should_terminate=False,
            expected_accuracy_drop=0.08226059,
        ),
        AccuracyDropTestCase(
            initial_metric=-0.2923,
            quantized_metric=-0.3185,
            drop_type=DropType.RELATIVE,
            expected_should_terminate=False,
            expected_accuracy_drop=0.0896339,
        ),
        AccuracyDropTestCase(
            initial_metric=-0.3185,
            quantized_metric=-0.2923,
            drop_type=DropType.RELATIVE,
            expected_should_terminate=True,
            expected_accuracy_drop=None,
        ),
    ],
)
def test_calculate_accuracy_drop(ts: AccuracyDropTestCase):
    should_terminate, accuracy_drop = calculate_accuracy_drop(
        ts.initial_metric, ts.quantized_metric, ts.max_drop, ts.drop_type
    )
    assert should_terminate == ts.expected_should_terminate
    assert pytest.approx(accuracy_drop) == ts.expected_accuracy_drop
