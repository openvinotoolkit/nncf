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

from typing import List, Optional

import numpy as np
import pytest

from nncf.common.tensor import NNCFTensor
from nncf.experimental.common.tensor_statistics.collectors import MergedTensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorAggregatorBase
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.common.tensor_statistics.collectors import TensorType


class DummyTensorReducer(TensorReducerBase):
    def __init__(self, output_name: str, inplace: bool = False, inplace_mock=None):
        super().__init__(inplace=inplace)
        self._output_name = output_name
        self._inplace_mock = inplace_mock

    def _reduce_out_of_place(self, x: List[TensorType]):
        return x

    def get_inplace_fn(self):
        return self._inplace_mock

    def get_output_names(self, target_node_name: str, port_id: int) -> str:
        return [self._output_name]

    def _get_processor(self):
        return None


class DummyTensorReducerA(DummyTensorReducer):
    pass


class DummyTensorAggregator(TensorAggregatorBase):
    def __init__(self, num_samples: Optional[int]):
        super().__init__(None, num_samples)

    def _register_reduced_input_impl(self, x: TensorType):
        return self._container.append(x)

    def aggregate(self):
        return self._container[0]


class DummyTensorAggregatorA(DummyTensorAggregator):
    pass


def test_aggregator_enabled_and_reset():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    input_name = "input_name"
    inputs = TensorCollector.get_tensor_collector_inputs(
        {input_name: NNCFTensor(np.array(100))}, [(hash(reducer), [input_name])]
    )

    for _ in range(3):
        collector.register_inputs(inputs)
    assert len(aggregator._container) == 3
    assert aggregator._collected_samples == 3

    collector.disable()

    for _ in range(3):
        collector.register_inputs(inputs)
    assert len(aggregator._container) == 3
    assert aggregator._collected_samples == 3

    collector.enable()

    for _ in range(3):
        collector.register_inputs(inputs)
    assert len(aggregator._container) == 5
    assert aggregator._collected_samples == 5

    collector.reset()
    assert len(aggregator._container) == 0
    assert aggregator._collected_samples == 0


def test_duplicated_statistics_are_merged():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    reducer_a = DummyTensorReducerA("A")
    keys = "ABC"
    aggregators = []
    for key in keys:
        aggregator = DummyTensorAggregator(5)
        collector.register_statistic_branch(key, reducer, aggregator)
        aggregators.append(aggregator)
    aggregator_a = DummyTensorAggregatorA(1)
    aggregator_b = DummyTensorAggregator(100)
    collector.register_statistic_branch("D", reducer, aggregator_a)
    collector.register_statistic_branch("E", reducer_a, aggregator_b)
    reducer_inplace = DummyTensorReducer("Dummy_inplace", True)
    aggregator_for_inplace = DummyTensorAggregator(4)
    collector.register_statistic_branch("F", reducer_inplace, aggregator_for_inplace)

    # Check reducers and aggregators are merged
    assert len(collector._reducers) == 3
    assert len(collector._aggregators) == 4
    assert collector.num_samples == 100

    output_info = collector.get_output_info(None, None)
    # Check output info
    assert sorted(output_info) == sorted(
        [(hash(reducer_inplace), ["Dummy_inplace"]), (hash(reducer_a), ["A"]), (hash(reducer), ["Dummy"])]
    )

    outputs = {"Dummy": NNCFTensor(np.array(5)), "A": NNCFTensor(np.array(0)), "Dummy_inplace": NNCFTensor(np.array(6))}
    target_inputs = TensorCollector.get_tensor_collector_inputs(outputs, output_info)
    collector.register_inputs(target_inputs)

    # Check aggregators recieved inputs as expected
    assert aggregators[0]._collected_samples == 1
    for aggregator in aggregators[1:]:
        assert aggregator._collected_samples == 0
    assert aggregator_a._collected_samples == 1
    assert aggregator_b._collected_samples == 1
    assert aggregator_for_inplace._collected_samples == 1

    statistics = collector.get_statistics()

    # Check aggregators recieved correct inputs
    assert len(statistics) == 6
    for k in "ABC":
        assert statistics[k] == NNCFTensor(np.array(5))
    assert statistics["D"] == NNCFTensor(np.array(5))
    assert statistics["E"] == NNCFTensor(np.array(0))
    assert statistics["F"] == NNCFTensor(np.array(6))


def test_inplace_param():
    inplace_op = lambda: 0
    collector = TensorCollector()
    reducer_out_of_place = DummyTensorReducer("Dummy")
    reducer_inplace = DummyTensorReducer("Dummy", True, inplace_op)
    reducer_other = DummyTensorReducerA("Dummy")
    aggregator_inplace = DummyTensorAggregator(5)
    aggregator_out_of_place = DummyTensorAggregator(5)
    aggregator_other = DummyTensorAggregator(5)

    collector.register_statistic_branch("out_of_place", reducer_out_of_place, aggregator_out_of_place)
    collector.register_statistic_branch("inplace", reducer_inplace, aggregator_inplace)
    collector.register_statistic_branch("other", reducer_other, aggregator_other)
    assert len(collector._reducers) == 3
    assert len(collector._aggregators) == 3
    assert collector.get_inplace_fn_info()[0][0] == inplace_op
    assert collector.any_stat_out_of_place()


def test_merged_tensor_collector():
    num_collectors = 4
    collectors = [TensorCollector() for _ in range(num_collectors)]
    for idx, collector in enumerate(collectors):
        reducer_common = DummyTensorReducer("common_input")
        aggregator_common = DummyTensorAggregator(5)
        reducer_unique = type(DummyTensorReducer.__name__ + str(idx), (DummyTensorReducer,), {})(f"input_{idx + 1}")
        aggregator_unique = type(DummyTensorAggregator.__name__ + str(idx), (DummyTensorAggregator,), {})(5)
        collector.register_statistic_branch("common", reducer_common, aggregator_common)
        collector.register_statistic_branch("unique", reducer_unique, aggregator_unique)

    collectors[-1].disable()
    merged_collector = MergedTensorCollector(collectors)

    # Check reducers and aggregators are merged correctly
    assert len(merged_collector._reducers) == num_collectors
    assert len(merged_collector._aggregators) == num_collectors

    # Check aggregators were replaced correctly
    common_branch_key = (hash(reducer_common), 0, hash(aggregator_common))
    common_aggregator = merged_collector._aggregators[common_branch_key]
    for collector in collectors[:-1]:
        assert collector.aggregators[common_branch_key] is common_aggregator

    output_info = merged_collector.get_output_info(None, None)
    outputs = {"common_input": NNCFTensor(np.array(0))}
    outputs.update({f"input_{idx + 1}": NNCFTensor(np.array(idx + 1)) for idx, _ in enumerate(collectors[:-1])})
    target_inputs = TensorCollector.get_tensor_collector_inputs(outputs, output_info)
    merged_collector.register_inputs(target_inputs)

    # Check statistics are collected in a correct way
    for idx, collector in enumerate(collectors[:-1]):
        for aggregator in collector._aggregators.values():
            assert aggregator._collected_samples == 1

        statistic = collector.get_statistics()
        assert len(statistic) == 2
        assert statistic["common"] == NNCFTensor(np.array(0))
        assert statistic["unique"] == NNCFTensor(np.array(idx + 1))


def test_ambigous_container_key():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    with pytest.raises(RuntimeError):
        collector.register_statistic_branch("A", reducer, aggregator)


def test_ambiguous_branches():
    collector = TensorCollector()
    reducer = DummyTensorReducer("Dummy")
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch("A", reducer, aggregator)
    with pytest.raises(RuntimeError):
        collector.register_statistic_branch("B", reducer, aggregator)


class DummyMultipleInpOutTensorReducer(DummyTensorReducer):
    NUM_INPUTS = 3
    NUM_OUTPUTS = 2

    def _reduce_out_of_place(self, x: List[TensorType]):
        return x[: self.NUM_OUTPUTS]

    def get_output_names(self, target_node_name: str, port_id: int) -> str:
        return [f"{target_node_name}_{port_id}_{self._output_name}_{i}" for i in range(self.NUM_INPUTS)]


def test_multiple_branch_reducer():
    reducer_output_name = "reducer_output_name"
    target_node_name = "target_node_name"
    collector = TensorCollector()
    reducer = DummyMultipleInpOutTensorReducer(reducer_output_name)

    for i in range(reducer.NUM_OUTPUTS):
        aggregator = DummyTensorAggregator(None)
        collector.register_statistic_branch(str(i), reducer, aggregator, i)

    ref_output_info = [
        (
            hash(reducer),
            [
                "target_node_name_0_reducer_output_name_0",
                "target_node_name_0_reducer_output_name_1",
                "target_node_name_0_reducer_output_name_2",
            ],
        )
    ]
    inputs = {name: NNCFTensor(np.array(i)) for i, name in enumerate(ref_output_info[0][1])}

    output_info = collector.get_output_info(target_node_name, 0)
    assert output_info == ref_output_info

    target_inputs = collector.get_tensor_collector_inputs(inputs, output_info)
    collector.register_inputs(target_inputs)

    ref_stats = {"0": NNCFTensor(np.array(0)), "1": NNCFTensor(np.array(1))}
    stats = collector.get_statistics()
    assert len(ref_stats) == len(stats)
    for key in ref_stats:
        assert ref_stats[key] == stats[key]
