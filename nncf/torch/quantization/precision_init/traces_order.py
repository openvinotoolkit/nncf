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
from typing import List

from torch import Tensor


class TracesOrder:
    def __init__(self, execution_indexes_of_weights_ordered_by_traces: List[int]):
        self._index_by_traces_to_execution_index = execution_indexes_of_weights_ordered_by_traces
        self._num_weights = len(execution_indexes_of_weights_ordered_by_traces)
        self._index_by_execution_to_index_by_traces = [
            execution_indexes_of_weights_ordered_by_traces.index(i) for i in range(self._num_weights)
        ]

    def get_execution_order_configs(self, trace_ordered_configuration: List) -> List:
        if len(trace_ordered_configuration) != self._num_weights:
            raise ValueError("Incompatible configuration size!")
        execution_order_config = [None] * self._num_weights
        for i, config in enumerate(trace_ordered_configuration):
            execution_order_config[self._index_by_traces_to_execution_index[i]] = config
        return execution_order_config

    def get_traces_order_configs(self, execution_ordered_configuration: List) -> List:
        if len(execution_ordered_configuration) != self._num_weights:
            raise ValueError("Incompatible configuration size!")
        traces_order_config = [None] * self._num_weights
        for i, config in enumerate(execution_ordered_configuration):
            traces_order_config[self._index_by_execution_to_index_by_traces[i]] = config
        return traces_order_config

    def get_execution_index_by_traces_index(self, traces_index: int):
        return self._index_by_traces_to_execution_index[traces_index]

    def __bool__(self):
        return bool(self._index_by_traces_to_execution_index)

    def __len__(self):
        return len(self._index_by_execution_to_index_by_traces)


class TracesPerLayer:
    def __init__(self, traces_per_layer_by_execution: Tensor):
        self._traces_per_layer_by_execution = traces_per_layer_by_execution
        execution_indexes_of_weights_in_descending_order_of_traces = [
            i[0] for i in sorted(enumerate(traces_per_layer_by_execution), reverse=False, key=lambda x: x[1])
        ]
        self.traces_order = TracesOrder(execution_indexes_of_weights_in_descending_order_of_traces)

    def get_by_execution_index(self, execution_index: int) -> Tensor:
        return self._traces_per_layer_by_execution[execution_index]

    def get_by_trace_index(self, trace_index: int) -> Tensor:
        execution_index = self.traces_order.get_execution_index_by_traces_index(trace_index)
        return self._traces_per_layer_by_execution[execution_index]

    def get_all(self) -> Tensor:
        return self._traces_per_layer_by_execution

    def __bool__(self):
        return bool(self.traces_order)
