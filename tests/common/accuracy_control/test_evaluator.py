# Copyright (c) 2024 Intel Corporation
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
import logging
from typing import List, TypeVar, Union

import numpy as np
import pytest

import nncf
from nncf.common.utils.backend import BackendType, get_available_backends
from nncf.data.dataset import Dataset
from nncf.quantization.algorithms.accuracy_control.evaluator import Evaluator, MetricResults
from nncf.quantization.algorithms.accuracy_control.onnx_backend import ONNXPreparedModel
from nncf.quantization.algorithms.accuracy_control.openvino_backend import (
    OVPreparedModel,
)
from tests.onnx.models import LinearModel as NxLinearModel
from tests.openvino.native.common import get_dataset_for_test
from tests.openvino.native.models import LinearModel as OvLinearModel
from tests.tensorflow.helpers import get_basic_conv_test_model

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
            _ = Evaluator.determine_mode(
                prepared_model, Dataset([None]), _validation_fn
            )
    else:
        is_metric_mode = Evaluator.determine_mode(
            prepared_model, Dataset([None]), _validation_fn
        )
        assert is_metric_mode == ts.expected_is_metric_mode


def test_determine_mode_2(mocker):
    def _validation_fn_with_error(dummy_model, dummy_dataset):
        raise RuntimeError

    prepared_model = mocker.Mock()
    prepared_model.model_for_inference = None

    is_metric_mode = Evaluator.determine_mode(
        prepared_model, Dataset([None]), _validation_fn_with_error
    )
    assert not is_metric_mode


@pytest.fixture
def evaluator():
    def _validation_fn(model, dataset):
        len = 0
        for i in dataset:
            len += 1

        return (0.1, [int(1)] * len)

    evaluator = Evaluator(_validation_fn)
    return evaluator


def test_evaluator_init(evaluator):
    assert evaluator.num_passed_iterations == 0
    assert evaluator.is_metric_mode() == None

    evaluator.enable_iteration_count()
    assert evaluator._enable_iteration_count == True

    evaluator.disable_iteration_count()
    assert evaluator._enable_iteration_count == False


@dataclass
class PrepareModeTestStruct:
    model: TModel
    type: BackendType


@pytest.mark.parametrize(
    "ts",
    [
        PrepareModeTestStruct(OvLinearModel().ov_model, OVPreparedModel),
        PrepareModeTestStruct(NxLinearModel().onnx_model, ONNXPreparedModel),
    ],
)
def test_prepare_mode_no_error(ts, evaluator, mocker):
    mocker.patch(
        "nncf.common.utils.backend.get_available_backends",
        return_value=[BackendType.ONNX, BackendType.OPENVINO],
    )
    result = evaluator.prepare_model(ts.model)
    assert isinstance(result, ts.type)


@dataclass
class PrepareModeErrorTestStruct:
    model: TModel
    available_backends: List[BackendType]
    error: Exception


@pytest.mark.parametrize(
    "ts",
    [
        PrepareModeErrorTestStruct(
            NxLinearModel().onnx_model,
            [BackendType.OPENVINO],
            nncf.UnsupportedBackendError,
        ),
        PrepareModeErrorTestStruct(
            get_basic_conv_test_model(),
            [BackendType.OPENVINO, BackendType.TENSORFLOW, BackendType.ONNX],
            NotImplementedError,
        ),
    ],
)
def test_prepare_mode_error(ts, evaluator, mocker):
    mocker.patch(
        "nncf.common.utils.backend.get_available_backends",
        return_value=ts.available_backends,
    )
    with pytest.raises(ts.error):
        evaluator.prepare_model(ts.model)


@pytest.fixture
def prepared_ov_model_and_data(evaluator, mocker):
    mocker.patch(
        "nncf.common.utils.backend.get_available_backends",
        return_value=[BackendType.OPENVINO],
    )
    model = OvLinearModel().ov_model
    return evaluator.prepare_model(model), get_dataset_for_test(model)


def test_validate_prepared_model_value_error(evaluator, prepared_ov_model_and_data):
    prepared_ov_model, dataset = prepared_ov_model_and_data
    evaluator._metric_mode = False
    with pytest.raises(ValueError):
        evaluator.validate_prepared_model(prepared_ov_model, dataset, [0, 1])


@pytest.mark.parametrize(
    "enable_iteration_count, metric_mode, expected_iterations, expected_item_types",
    [(False, False, 0, int), (True, True, 1, float)],
)
def test_validate(
    evaluator,
    mocker,
    enable_iteration_count,
    metric_mode,
    expected_iterations,
    expected_item_types,
):
    if enable_iteration_count:
        evaluator.enable_iteration_count()
    else:
        evaluator.disable_iteration_count()
    evaluator._metric_mode = metric_mode
    mocker.patch(
        "nncf.common.utils.backend.get_available_backends",
        return_value=[BackendType.ONNX, BackendType.OPENVINO],
    )
    model = OvLinearModel().ov_model
    metric, values_for_each_item = evaluator.validate(
        model, get_dataset_for_test(model)
    )
    assert evaluator.num_passed_iterations == expected_iterations
    assert isinstance(metric, float)
    for f in values_for_each_item:
        assert isinstance(f, expected_item_types)


@pytest.mark.parametrize(
    "metric_mode, enable_iteration_count, expected_item_type, expected_iterations",
    [(False, True, list, 1), (True, False, float, 0)],
)
def test_collect_values_for_each_item(
    evaluator,
    metric_mode,
    enable_iteration_count,
    expected_item_type,
    expected_iterations,
    mocker,
):
    if enable_iteration_count:
        evaluator.enable_iteration_count()
    else:
        evaluator.disable_iteration_count()
    evaluator._metric_mode = metric_mode

    mocker.patch(
        "nncf.common.utils.backend.get_available_backends",
        return_value=[BackendType.ONNX, BackendType.OPENVINO],
    )

    model = OvLinearModel().ov_model
    result = evaluator.collect_values_for_each_item(model, get_dataset_for_test(model))

    assert all(isinstance(e, expected_item_type) for e in result)
    assert evaluator._num_passed_iterations == expected_iterations


def test_collect_metric_results(evaluator, mocker, nncf_caplog):
    mocker.patch(
        "nncf.common.utils.backend.get_available_backends",
        return_value=[BackendType.ONNX, BackendType.OPENVINO],
    )

    model = OvLinearModel().ov_model

    result = evaluator.collect_metric_results(model, get_dataset_for_test(model), "test_model")
    assert isinstance(result, MetricResults)

    with nncf_caplog.at_level(logging.INFO):
        assert "Validation of test_model model was started" in nncf_caplog.text
        assert "Metric of test_model model: 0.1" in nncf_caplog.text
