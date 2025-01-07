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
from abc import ABC
from abc import abstractmethod
from typing import List

import numpy as np
import pytest

from nncf.common.graph.utils import get_reduction_axes
from nncf.experimental.common.tensor_statistics.collectors import TensorCollector
from nncf.experimental.common.tensor_statistics.statistics import MinMaxTensorStatistic
from nncf.tensor import Tensor
from nncf.tensor import functions as fns


class TemplateTestTensorCollectorBatchSize(ABC):
    @pytest.fixture
    @abstractmethod
    def reducers(self):
        pass

    @pytest.fixture
    @abstractmethod
    def aggregators(self):
        pass

    @pytest.fixture
    @abstractmethod
    def inplace(self):
        pass

    @staticmethod
    @abstractmethod
    def to_backend_tensor(self, tensor: np.ndarray):
        pass

    def create_dataitems_without_batch_dim(self, input_shape: List[int], length: int = 100) -> List[np.ndarray]:
        rng = np.random.default_rng(seed=0)
        data_items = []
        for _ in range(length):
            data_items.append(rng.uniform(0, 1, input_shape))
        return data_items

    def add_batch_dim_to_dataitems(self, data_items: List[np.ndarray], batch_size: int) -> List[np.ndarray]:
        assert batch_size >= 1
        dataset = []
        item = []
        cnt = 0
        for data_item in data_items:
            if batch_size == 1:
                dataset.append(np.expand_dims(data_item, 0))
            else:
                item.append(data_item)
                if cnt == batch_size - 1:
                    dataset.append(np.array(item))
                    item = []
                    cnt = -1
                cnt += 1

        return dataset

    def _create_tensor_collector(self, shape, inplace, reducer, aggregator) -> TensorCollector:
        batch_axis = 0
        statistic_branch_random_name = "1"
        collector = TensorCollector(MinMaxTensorStatistic)
        reduction_axes = get_reduction_axes([batch_axis], shape)
        aggregation_axes = (0, 1)
        kwargs = {"reduction_axes": reduction_axes, "inplace": inplace}
        reducer = reducer(**kwargs)
        aggregator = aggregator(
            aggregation_axes=aggregation_axes,
        )
        collector.register_statistic_branch(statistic_branch_random_name, reducer, aggregator)
        return collector, reducer, aggregator

    def _register_inputs(self, collector, dataitems, reducer):
        for item in dataitems:
            item = self.to_backend_tensor(item)
            input_ = {hash(reducer): [Tensor(item)]}
            collector.register_inputs(input_)

    def test_statistics_batch_size_equal(self, reducers, aggregators, inplace):
        tensor_shape = [3, 20, 20]
        dataitems = self.create_dataitems_without_batch_dim(input_shape=tensor_shape)

        shape_batch_1 = [1, *tensor_shape]
        collector, reducer, _ = self._create_tensor_collector(shape_batch_1, inplace, reducers, aggregators)
        dataitems_batch_1 = self.add_batch_dim_to_dataitems(dataitems, batch_size=1)
        self._register_inputs(collector, dataitems_batch_1, reducer)
        aggregated_tensor_batch_1 = list(collector._aggregate().values())

        shape_batch_10 = [10, *tensor_shape]
        collector, reducer, _ = self._create_tensor_collector(shape_batch_10, inplace, reducers, aggregators)
        dataitems_batch_10 = self.add_batch_dim_to_dataitems(dataitems, batch_size=10)
        self._register_inputs(collector, dataitems_batch_10, reducer)
        aggregated_tensor_batch_10 = list(collector._aggregate().values())

        assert fns.allclose(fns.stack(aggregated_tensor_batch_1), fns.stack(aggregated_tensor_batch_10))
