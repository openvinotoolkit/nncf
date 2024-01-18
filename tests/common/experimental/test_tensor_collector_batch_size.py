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
from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np
import pytest

from nncf.common.graph.utils import get_channel_agnostic_reduction_axes
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector


class TemplateTestTensorCollectorBatchSize(ABC):
    @staticmethod
    @abstractmethod
    def get_tensor_statistics_class():
        ...

    @staticmethod
    @abstractmethod
    def get_tensor_processor():
        ...

    @staticmethod
    @abstractmethod
    def get_nncf_tensor_class():
        ...

    @pytest.fixture
    @abstractmethod
    def reducers(self):
        ...

    @pytest.fixture
    @abstractmethod
    def aggregators(self):
        ...

    @pytest.fixture
    @abstractmethod
    def inplace(self):
        ...

    @abstractmethod
    def create_dataitems_without_batch_dim(self, input_shape: List[int], length: int = 100) -> List[np.ndarray]:
        ...

    @abstractmethod
    def add_batch_dim_to_dataitems(self, data_items: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        ...

    def _create_tensor_collector(self, shape, inplace, reducer, aggregator) -> TensorCollector:
        batch_axis = 0
        statistic_branch_random_name = "1"
        collector = TensorCollector(self.get_tensor_statistics_class())
        reduction_axes = get_channel_agnostic_reduction_axes([batch_axis], shape)
        aggregation_axes = (0, 1)
        kwargs = {"reduction_axes": reduction_axes, "inplace": inplace}
        reducer = reducer(**kwargs)
        aggregator = aggregator(
            aggregation_axes=aggregation_axes,
            tensor_processor=self.get_tensor_processor(),
        )
        collector.register_statistic_branch(statistic_branch_random_name, reducer, aggregator)
        return collector, reducer, aggregator

    def _register_inputs(self, collector, dataitems, reducer):
        for item in dataitems:
            input_ = {hash(reducer): [self.get_nncf_tensor_class()(item)]}
            collector.register_inputs(input_)

    def test_statistics_batch_size_equal(self, reducers, aggregators, inplace):
        tensor_shape = [3, 20, 20]
        dataitems = self.create_dataitems_without_batch_dim(input_shape=tensor_shape)

        shape_batch_1 = [1, *tensor_shape]
        collector, reducer, _ = self._create_tensor_collector(shape_batch_1, inplace, reducers, aggregators)
        # output_name = reducer.get_output_names(target_node_name, port_id)
        dataitems_batch_1 = self.add_batch_dim_to_dataitems(dataitems, batch_size=1)
        self._register_inputs(collector, dataitems_batch_1, reducer)
        aggregated_tensor_batch_1 = list(collector._aggregate().values())

        shape_batch_10 = [10, *tensor_shape]
        collector, reducer, _ = self._create_tensor_collector(shape_batch_10, inplace, reducers, aggregators)
        # output_name = reducer.get_output_names(target_node_name, port_id)
        dataitems_batch_10 = self.add_batch_dim_to_dataitems(dataitems, batch_size=10)
        self._register_inputs(collector, dataitems_batch_10, reducer)
        aggregated_tensor_batch_10 = list(collector._aggregate().values())

        assert np.array_equal(aggregated_tensor_batch_1, aggregated_tensor_batch_10)
