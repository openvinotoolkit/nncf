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

from typing import Any, Iterable, List, Tuple

import numpy as np
import openvino.runtime as ov
from openvino.tools import pot

from nncf.data.dataset import Dataset
from nncf.openvino.pot.engine import OVEngine
from nncf.openvino.pot.engine import calc_per_sample_metrics
from tests.openvino.native.models import LinearModel
from tests.openvino.pot.utils import convert_openvino_model_to_compressed_model


class DummySampler:
    def __init__(self, indices: List[int]):
        self._subset_indices = indices


class DummyDataset:
    def __init__(self, shape: Tuple[int]):
        self._length = 5
        self._data = [np.random.rand(*shape).astype(np.float32) for _ in range(self._length)]

    def __iter__(self):
        return iter(self._data)


def val_func(compiled_model: ov.CompiledModel, dataset: Iterable[Any]) -> float:
    output = compiled_model.output(0)

    values = []
    for inputs in dataset:
        predictions = compiled_model([inputs])[output]
        values.append(np.sum(predictions))

    return np.mean(values).item()


def get_expected(model: ov.Model, dataset, use_output: bool = False):
    compiled_model = ov.Core().compile_model(model, device_name="CPU")
    metric = val_func(compiled_model, dataset)

    per_sample_metrics = []
    output = compiled_model.output(0)
    for idx, data_item in enumerate(dataset):
        if use_output:
            value = compiled_model([data_item])[output]
        else:
            value = val_func(compiled_model, [data_item])

        per_sample_metrics.append({"sample_id": idx, "result": value})
    return per_sample_metrics, metric


def test_predict_original_metric():
    subset_indices = [0, 2, 4]
    dataset = Dataset(DummyDataset(shape=(1, 3, 4, 2)))

    ov_model = LinearModel().ov_model
    expected_per_sample, expected_metric = get_expected(ov_model, dataset.get_data(subset_indices), use_output=False)

    pot_model = convert_openvino_model_to_compressed_model(ov_model, target_device="CPU")
    engine = OVEngine({"device": "CPU"}, dataset, dataset, val_func, use_original_metric=True)
    engine.set_model(pot_model)

    (actual_per_sample, actual_metric), _ = engine.predict(sampler=DummySampler(subset_indices), metric_per_sample=True)

    assert np.allclose(expected_metric, actual_metric["original_metric"])
    for expected, actual in zip(expected_per_sample, actual_per_sample):
        assert expected["sample_id"] == actual["sample_id"]
        assert np.allclose(expected["result"], actual["result"])


def test_predict_output():
    dataset = Dataset(DummyDataset(shape=(1, 3, 4, 2)))

    ov_model = LinearModel().ov_model
    expected_per_sample, expected_metric = get_expected(ov_model, dataset.get_data(), use_output=True)
    pot_model = convert_openvino_model_to_compressed_model(ov_model, target_device="CPU")
    engine = OVEngine({"device": "CPU"}, dataset, dataset, val_func, use_original_metric=False)
    engine.set_model(pot_model)

    stats_layout = {}
    stats_layout["MatMul"] = {"output_logits": pot.statistics.statistics.TensorStatistic(lambda a: a)}

    (actual_per_sample, actual_metric), raw_output = engine.predict(stats_layout, metric_per_sample=True)

    raw_output = raw_output["MatMul"]["output_logits"]
    for idx, data in enumerate(actual_per_sample):
        data["result"] = raw_output[idx]

    assert np.allclose(expected_metric, actual_metric["original_metric"])
    for expected, actual in zip(expected_per_sample, actual_per_sample):
        assert expected["sample_id"] == actual["sample_id"]
        assert np.allclose(expected["result"], actual["result"])


def test_calc_per_sample():
    dataset = Dataset(iter(DummyDataset(shape=(1, 3, 4, 2))))  # Can iterate only once
    ov_model = LinearModel().ov_model
    compiled_model = ov.Core().compile_model(ov_model, device_name="CPU")

    # Check that we iterate through dataset only once during
    # per-sample metrics calculation.
    _ = calc_per_sample_metrics(compiled_model, val_func, dataset, [0, 2, 4])
