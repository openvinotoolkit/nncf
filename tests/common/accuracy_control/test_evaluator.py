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

import logging
from dataclasses import dataclass
from typing import List, TypeVar, Union
from unittest.mock import MagicMock
from unittest.mock import Mock

import numpy as np
import pytest

import nncf
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator
from nncf.quantization.algorithms.accuracy_control.evaluator import MetricResults

TModel = TypeVar("TModel")


@dataclass
class ModeTestStruct:
    metric_value: float
    values_for_each_item: Union[None, List[float], List[List[np.ndarray]]]
    expected_is_metric_mode: bool
    raise_exception: bool = False


@pytest.mark.parametrize(
    "ts",
    [
        # Return: (float, None)
        ModeTestStruct(
            metric_value=0.1,
            values_for_each_item=None,
            expected_is_metric_mode=True,
        ),
        # Return: (float, List[float])
        ModeTestStruct(
            metric_value=0.1,
            values_for_each_item=[0.3],
            expected_is_metric_mode=True,
        ),
        # Return: (float, List[List[TTensor]])
        ModeTestStruct(
            metric_value=0.1,
            values_for_each_item=[[np.array([1.1])]],
            expected_is_metric_mode=False,
        ),
        # Return: (None, None)
        ModeTestStruct(
            metric_value=None,
            values_for_each_item=None,
            expected_is_metric_mode=False,
        ),
        # Return: (None, List[float])
        ModeTestStruct(
            metric_value=None,
            values_for_each_item=[0.3],
            expected_is_metric_mode=None,
            raise_exception=True,
        ),
        # Return: (None, List[List[TTensor]])
        ModeTestStruct(
            metric_value=None,
            values_for_each_item=[[np.array([1.1])]],
            expected_is_metric_mode=False,
        ),
        # Return: (ConvertibleToFloat, List[ConvertibleToFloat])
        ModeTestStruct(
            metric_value=np.array(0.1),
            values_for_each_item=[np.array(0.3)],
            expected_is_metric_mode=True,
        ),
        # Return: (ConvertibleToFloat, List[ConvertibleToFloat])
        ModeTestStruct(
            metric_value=np.array([0.1]),
            values_for_each_item=[np.array([0.3])],
            expected_is_metric_mode=True,
        ),
        # Return: (NotConvertibleToFloat, None)
        ModeTestStruct(
            metric_value=[0.1],
            values_for_each_item=None,
            expected_is_metric_mode=None,
            raise_exception=True,
        ),
    ],
)
def test_determine_mode(ts: ModeTestStruct, mocker):
    def _validation_fn(dummy_model, dummy_dataset):
        return (ts.metric_value, ts.values_for_each_item)

    prepared_model = mocker.Mock()
    prepared_model.model_for_inference = None

    if ts.raise_exception:
        with pytest.raises(nncf.InternalError):
            _ = Evaluator.determine_mode(prepared_model, Dataset([None]), _validation_fn)
    else:
        is_metric_mode = Evaluator.determine_mode(prepared_model, Dataset([None]), _validation_fn)
        assert is_metric_mode == ts.expected_is_metric_mode


def test_determine_mode_2(mocker):
    def _validation_fn_with_error(dummy_model, dummy_dataset):
        raise RuntimeError

    prepared_model = mocker.Mock()
    prepared_model.model_for_inference = None

    is_metric_mode = Evaluator.determine_mode(prepared_model, Dataset([None]), _validation_fn_with_error)
    assert not is_metric_mode


@pytest.fixture
def evaluator_returns_list_of_float():
    def _validation_fn(model, dataset):
        for _ in dataset:
            pass

        return (0.1, [0.1])

    evaluator = Evaluator(_validation_fn)
    return evaluator


@pytest.fixture
def evaluator_returns_tensor():
    def _validation_fn(model, dataset):
        for _ in dataset:
            pass

        return (0.1, [[np.array([1.1], dtype=np.float32)]])

    evaluator = Evaluator(_validation_fn)
    return evaluator


def test_evaluator_init(evaluator_returns_list_of_float):
    evaluator = evaluator_returns_list_of_float
    assert evaluator.num_passed_iterations == 0
    assert evaluator.is_metric_mode() is None

    evaluator.enable_iteration_count()
    assert evaluator._enable_iteration_count

    evaluator.disable_iteration_count()
    assert not evaluator._enable_iteration_count


@pytest.fixture
def model_and_data():
    prepared_model = MagicMock()
    dataset = nncf.Dataset([1, 1, 1, 1])
    return prepared_model, dataset


# `indices` parameter can be used only if Evaluator.is_metric_mode
#  raise ValueError if Evaluator is not in metric mode and indices are passed to validate_prepared_model
def test_validate_prepared_model_value_error(evaluator_returns_list_of_float, model_and_data):
    prepared_ov_model, dataset = model_and_data
    evaluator_returns_list_of_float._metric_mode = False
    with pytest.raises(ValueError):
        evaluator_returns_list_of_float.validate_prepared_model(prepared_ov_model, dataset, [0, 1])


@pytest.mark.parametrize(
    "enable_iteration_count, expected_iterations",
    [(False, 0), (True, 4)],
)
def test_validate_metric_mode(
    evaluator_returns_list_of_float, model_and_data, mocker, enable_iteration_count, expected_iterations
):
    evaluator_returns_list_of_float._metric_mode = None
    if enable_iteration_count:
        evaluator_returns_list_of_float.enable_iteration_count()

    model, dataset = model_and_data
    mocker.patch("nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=model)
    metric, values_for_each_item = evaluator_returns_list_of_float.validate(model, dataset)
    assert all(isinstance(item, float) for item in values_for_each_item)
    assert evaluator_returns_list_of_float.num_passed_iterations == expected_iterations

    assert isinstance(metric, float)


@pytest.mark.parametrize(
    "enable_iteration_count, expected_iterations, metric_mode",
    [(False, 0, None), (True, 4, None), (False, 0, False), (True, 4, False)],
)
def test_validate_metric_mode_none_or_false(
    evaluator_returns_tensor, model_and_data, mocker, enable_iteration_count, expected_iterations, metric_mode
):
    evaluator_returns_tensor._metric_mode = metric_mode
    if enable_iteration_count:
        evaluator_returns_tensor.enable_iteration_count()

    model, dataset = model_and_data
    mocker.patch("nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=model)

    metric, values_for_each_item = evaluator_returns_tensor.validate(model, dataset)
    assert evaluator_returns_tensor.num_passed_iterations == expected_iterations
    assert all(isinstance(item, List) for item in values_for_each_item)
    assert all(isinstance(item, np.ndarray) for item in values_for_each_item[0])
    assert isinstance(metric, float)


@pytest.mark.parametrize(
    "metric_mode, enable_iteration_count, expected_item_type, expected_iterations",
    [(False, True, list, 4), (False, False, list, 0), (True, True, float, 4), (True, False, float, 0)],
)
def test_collect_values_for_each_item(
    evaluator_returns_list_of_float,
    metric_mode,
    enable_iteration_count,
    expected_item_type,
    expected_iterations,
    mocker,
    model_and_data,
):
    evaluator = evaluator_returns_list_of_float
    if enable_iteration_count:
        evaluator.enable_iteration_count()
    else:
        evaluator.disable_iteration_count()
    evaluator._metric_mode = metric_mode

    model, dataset = model_and_data
    model.model_for_inference = Mock()
    model.__call__ = Mock(return_value=2)
    mocker.patch("nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=model)

    result = evaluator.collect_values_for_each_item(model, dataset)
    assert all(isinstance(e, expected_item_type) for e in result)
    assert evaluator._num_passed_iterations == expected_iterations


def test_collect_metric_results(model_and_data, evaluator_returns_list_of_float, mocker, nncf_caplog):
    model, dataset = model_and_data
    mocker.patch("nncf.quantization.algorithms.accuracy_control.evaluator.Evaluator.prepare_model", return_value=model)

    result = evaluator_returns_list_of_float.collect_metric_results(model, dataset, "test_model")
    assert isinstance(result, MetricResults)
    with nncf_caplog.at_level(logging.INFO):
        assert "Validation of test_model model was started" in nncf_caplog.text
        assert "Metric of test_model model: 0.1" in nncf_caplog.text
