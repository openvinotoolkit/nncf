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

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import pytest

from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator


class DummyAccuracyControlAlgoBackend:
    @staticmethod
    def prepare_for_inference(model):
        return model


@dataclass
class TestCase:
    metric_value: float
    values_for_each_item: Union[None, List[float], List[List[np.ndarray]]]
    expected_is_metric_mode: bool
    raise_exception: bool = False


@pytest.mark.parametrize(
    "ts",
    [
        # Return: (float, None)
        TestCase(
            metric_value=0.1,
            values_for_each_item=None,
            expected_is_metric_mode=True,
        ),
        # Return: (float, List[float])
        TestCase(
            metric_value=0.1,
            values_for_each_item=[0.3],
            expected_is_metric_mode=True,
        ),
        # Return: (float, List[List[TTensor]])
        TestCase(
            metric_value=0.1,
            values_for_each_item=[[np.array([1.1])]],
            expected_is_metric_mode=False,
        ),
        # Return: (None, None)
        TestCase(
            metric_value=None,
            values_for_each_item=None,
            expected_is_metric_mode=False,
        ),
        # Return: (None, List[float])
        TestCase(
            metric_value=None,
            values_for_each_item=[0.3],
            expected_is_metric_mode=None,
            raise_exception=True,
        ),
        # Return: (None, List[List[TTensor]])
        TestCase(metric_value=None, values_for_each_item=[[np.array([1.1])]], expected_is_metric_mode=False),
        # Return: (ConvertibleToFloat, List[ConvertibleToFloat])
        TestCase(
            metric_value=np.array(0.1),
            values_for_each_item=[np.array(0.3)],
            expected_is_metric_mode=True,
        ),
        # Return: (ConvertibleToFloat, List[ConvertibleToFloat])
        TestCase(
            metric_value=np.array([0.1]),
            values_for_each_item=[np.array([0.3])],
            expected_is_metric_mode=True,
        ),
        # Return: (NotConvertibleToFloat, None)
        TestCase(metric_value=[0.1], values_for_each_item=None, expected_is_metric_mode=None, raise_exception=True),
    ],
)
def test_determine_mode(ts: TestCase):
    def _validation_fn(dummy_model, dummy_dataset):
        return (ts.metric_value, ts.values_for_each_item)

    evaluator = Evaluator(_validation_fn, DummyAccuracyControlAlgoBackend())

    # pylint: disable=W0212
    if ts.raise_exception:
        with pytest.raises(RuntimeError):
            evaluator._determine_mode(None, Dataset([None]))
    else:
        evaluator._determine_mode(None, Dataset([None]))
        assert evaluator.is_metric_mode() == ts.expected_is_metric_mode
