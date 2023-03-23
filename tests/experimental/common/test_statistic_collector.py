import pytest
from typing import Optional
import numpy as np

from nncf.experimental.common.tensor_statistics.collectors import TensorType
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.collectors import TensorReducerBase
from nncf.experimental.common.tensor_statistics.collectors import TensorAggregatorBase
from nncf.experimental.common.tensor_statistics.collectors import MergedTensorCollector


class DummyTensorReducer(TensorReducerBase):
    def __init__(self, output_name: str, inplace: bool = False,
                 inplace_mock = None):
        super().__init__(inplace=inplace)
        self._output_name = output_name
        self._inplace_mock = inplace_mock

    def _reduce_out_of_place(self, x: TensorType):
        return x

    def get_inplace_fn(self):
        return self._inplace_mock

    def get_output_name(self, target_node_name: str, port_id: int) -> str:
        return self._output_name

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


def test_aggregator_enabled():
    collector = TensorCollector()
    reducer = DummyTensorReducer('Dummy')
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch('A', reducer, aggregator)
    inputs = {collector.get_output_info(None, None)[0][0]: np.array(100)}

    for _ in range(3):
        collector.register_inputs(inputs)
    assert aggregator._collected_samples == 3

    collector.disable()

    for _ in range(3):
        collector.register_inputs(inputs)
    assert aggregator._collected_samples == 3

    collector.enable()

    for _ in range(3):
        collector.register_inputs(inputs)
    assert aggregator._collected_samples == 5


def test_duplicated_statistics_are_merged():
    collector = TensorCollector()
    reducer = DummyTensorReducer('Dummy')
    reducer_a = DummyTensorReducerA('A')
    keys = 'ABC'
    aggregators = []
    for key in keys:
        aggregator = DummyTensorAggregator(5)
        collector.register_statistic_branch(key, reducer, aggregator)
        aggregators.append(aggregator)
    aggregator_a = DummyTensorAggregatorA(1)
    aggregator_b = DummyTensorAggregator(100)
    collector.register_statistic_branch('D', reducer, aggregator_a)
    collector.register_statistic_branch('E', reducer_a, aggregator_b)

    # Check reducers and aggregators are merged
    assert len(collector._reducers) == 2
    assert len(collector._aggregators) == 3
    assert collector.num_samples == 100

    output_info = collector.get_output_info(None, None)
    # Check output info
    assert sorted(output_info) == [('DummyTensorReducer', 'Dummy'), ('DummyTensorReducerA', 'A')]

    outputs = {'Dummy': np.array(5), 'A': np.array(0)}
    collector.register_inputs({reducer: outputs[name] for reducer, name in output_info})

    # Check aggregators recieved inputs as expected
    assert aggregators[0]._collected_samples == 1
    for aggregator in aggregators[1:]:
        assert aggregator._collected_samples == 0
    assert aggregator_a._collected_samples == 1
    assert aggregator_b._collected_samples == 1

    statistics = collector.get_statistics()

    # Check aggregators recieved correct inputs
    assert len(statistics) == 5
    for k in 'ABC':
        assert statistics[k] == np.array(5)
    assert statistics['D'] == np.array(5)
    assert statistics['E'] == np.array(0)


def test_inplace_param():
    inplace_op = lambda: 0
    collector = TensorCollector()
    reducer_inplace = DummyTensorReducer('Dummy', True, inplace_op)
    reducer_out_of_place = DummyTensorReducerA('Dummy')
    aggregator_inplace = DummyTensorAggregator(5)
    aggregator_out_of_place = DummyTensorAggregator(5)

    collector.register_statistic_branch('inplace', reducer_inplace, aggregator_inplace)
    collector.register_statistic_branch('out_of_place', reducer_out_of_place, aggregator_out_of_place)
    assert len(collector._reducers) == 2
    assert len(collector._aggregators) == 2
    assert collector.get_inplace_fn_info()[0] == inplace_op
    assert collector.any_stat_out_of_place()


def test_merged_tensor_collector():
    num_collectors = 4
    collectors = [TensorCollector() for _ in range(num_collectors)]
    for idx, collector in enumerate(collectors):
        reducer_common = DummyTensorReducer('common_input')
        aggregator_common = DummyTensorAggregator(5)
        reducer_unique = type(DummyTensorReducer.__name__ + str(idx),
                              (DummyTensorReducer,), {})(f'input_{idx + 1}')
        aggregator_unique = type(DummyTensorAggregator.__name__ + str(idx),
                                 (DummyTensorAggregator, ), {})(5)
        collector.register_statistic_branch('common', reducer_common, aggregator_common)
        collector.register_statistic_branch('unique', reducer_unique, aggregator_unique)

    collectors[-1].disable()
    merged_collector = MergedTensorCollector(collectors)

    # Check reducers and aggregators are merged correctly
    assert len(merged_collector._reducers) == num_collectors
    assert len(merged_collector._aggregators) == num_collectors

    # Check aggregators was replaced correctly
    common_branch_key = (hash(reducer_common), hash(aggregator_common))
    common_aggregator = merged_collector._aggregators[common_branch_key]
    for collector in collectors[:-1]:
        assert collector.aggregators[common_branch_key] is common_aggregator

    output_info = merged_collector.get_output_info(None, None)
    outputs = {'common_input': np.array(0)}
    outputs.update({f'input_{idx + 1}': np.array(idx + 1) for idx, _ in enumerate(collectors[:-1])})
    merged_collector.register_inputs({reducer: outputs[name] for reducer, name in output_info})

    # Check statistics are collected in a correct way
    for idx, collector in enumerate(collectors[:-1]):
        for aggregator in collector._aggregators.values():
            assert aggregator._collected_samples == 1

        statistic = collector.get_statistics()
        assert len(statistic) == 2
        assert statistic['common'] == np.array(0)
        assert statistic['unique'] == np.array(idx + 1)


def test_ambigous_container_key():
    collector = TensorCollector()
    reducer = DummyTensorReducer('Dummy')
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch('A', reducer, aggregator)
    with pytest.raises(RuntimeError):
        collector.register_statistic_branch('A', reducer, aggregator)


def test_ambiguous_branches():
    collector = TensorCollector()
    reducer = DummyTensorReducer('Dummy')
    reducer_a = DummyTensorReducerA('Dummy')
    aggregator = DummyTensorAggregator(5)
    collector.register_statistic_branch('A', reducer, aggregator)
    with pytest.raises(RuntimeError):
        collector.register_statistic_branch('B', reducer, aggregator)
