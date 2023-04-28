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

from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.openvino.tensor import OVNNCFTensor
from tests.experimental.common.test_statistic_collector import DummyTensorAggregator
from tests.experimental.common.test_statistic_collector import DummyTensorReducer


# pylint:disable=protected-access
def test_empty_tensors_register():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    input_name = "input_name"
    full_inputs = TensorCollector.get_tensor_collector_inputs(
        {input_name: OVNNCFTensor(np.array([100]))}, [(hash(reducer), [input_name])]
    )
    empty_inputs = TensorCollector.get_tensor_collector_inputs(
        {input_name: OVNNCFTensor(np.array([]))}, [(hash(reducer), [input_name])]
    )

    for inputs in [full_inputs, empty_inputs, full_inputs]:
        collector.register_inputs(inputs)
    assert len(aggregator._container) == 2
    assert aggregator._collected_samples == 2


# pylint:disable=protected-access
def test_empty_inplace_tensors_register():
    collector = TensorCollector()
    inplace_reducer = DummyTensorReducer("Dummy", True)
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", inplace_reducer, aggregator)
    input_name = "input_name"
    full_inputs = TensorCollector.get_tensor_collector_inputs(
        {input_name: OVNNCFTensor(np.array([100]))}, [(hash(inplace_reducer), [input_name])]
    )
    empty_inputs = TensorCollector.get_tensor_collector_inputs(
        {input_name: OVNNCFTensor(np.array([]))}, [(hash(inplace_reducer), [input_name])]
    )

    for inputs in [full_inputs, empty_inputs, full_inputs]:
        collector.register_inputs(inputs)
    assert len(aggregator._container) == 2
    assert aggregator._collected_samples == 2
